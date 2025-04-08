import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import webdataset as wds
from torch import optim
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from presto import Presto
from presto.dataops import BANDS_GROUPS_IDX, MASK_STRATEGIES, MaskParams, plot_masked
from presto.dataops.dataset import (
    TAR_BUCKET,
    FranceCropsFullDataset,
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)
from presto.eval import (
    AlgaeBloomsEval,
    CropHarvestEval,
    CropHarvestMultiClassValidation,
    EuroSatEval,
    EvalTask,
    FuelMoistureEval,
)
from presto.model import LossWrapper, adjust_learning_rate, param_groups_weight_decay
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    device,
    initialize_logging,
    seed_everything,
    timestamp_dirname,
    update_data_dir,
)

logger = logging.getLogger("__main__")
os.environ["GOOGLE_CLOUD_PROJECT"] = "large-earth-model"

sys.argv = [
    'train.py',  # placeholder for script name
    '--train_url', 'data/dw_144_mini_shard_44.tar',
    '--val_url', 'data/dw_144_mini_shard_44.tar',
    '--val_per_n_steps', '1',
    '--cropharvest_per_n_validations', '0',
    '--skip_finetuning'
]
__file__ = 'train.py'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Parent directory to save output to, <output_dir>/wandb/ "
    "and <output_dir>/output/ will be written to. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Data is stored in <data_dir>/data. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument("--n_epochs", type=int, default=20)
argparser.add_argument("--val_per_n_steps", type=int, default=1000)
argparser.add_argument(
    "--cropharvest_per_n_validations",
    type=int,
    default=10,
    help="0 to skip cropharvest validation",
)
argparser.add_argument(
    "--cropharvest_val_n_per_class",
    type=int,
    default=-1,
    help="-1 for no limit",
)
argparser.add_argument("--max_learning_rate", type=float, default=0.001)
argparser.add_argument("--min_learning_rate", type=float, default=0.0)
argparser.add_argument("--warmup_epochs", type=int, default=2)

argparser.add_argument("--weight_decay", type=float, default=0.05)
argparser.add_argument(
    "--dynamic_world_loss_weight",
    type=float,
    default=2,
    help="Each dynamic world instance we be weighted by this amount relative to each eo instance",
)
argparser.add_argument("--batch_size", type=int, default=4096)
argparser.add_argument(
    "--dataloader_length", type=int, default=5950, help="-1 to re-estimate dataloader length"
)
argparser.add_argument(
    "--mask_strategies",
    type=str,
    default=[
        "group_bands",
        "random_timesteps",
        "chunk_timesteps",
        "random_combinations",
    ],
    nargs="+",
    help="`all` will use all available masking strategies (including single bands)",
)
argparser.add_argument("--mask_ratio", type=float, default=0.75)
argparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_plots", type=int, default=3)
argparser.add_argument("--wandb_org", type=str, default="nasa-harvest")

argparser.add_argument(
    "--train_url",
    type=str,
    default=f"gs://{TAR_BUCKET}/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/"
    + "dw_144_shard_{0..58}.tar",
)
argparser.add_argument(
    "--val_url",
    type=str,
    default=f"gs://{TAR_BUCKET}/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/"
    + "dw_144_shard_59.tar",
)
argparser.add_argument("--skip_finetuning", dest="skip_finetuning", action="store_true")

argparser.set_defaults(wandb=False)
argparser.set_defaults(skip_finetuning=False)
args = argparser.parse_args().__dict__

model_name = args["model_name"]
seed: int = args["seed"]
path_to_config = args["path_to_config"]
wandb_enabled: bool = args["wandb"]
wandb_plots: int = args["wandb_plots"]
wandb_org: str = args["wandb_org"]

seed_everything(seed)

output_parent_dir = Path(args["output_dir"]) if args["output_dir"] else Path(__file__).parent
run_id = None
if wandb_enabled:
    import wandb

    run = wandb.init(
        entity=wandb_org,
        project="lem",
        dir=output_parent_dir,
    )
    run_id = cast(Run, run).id

logging_dir = output_parent_dir / "output" / timestamp_dirname(run_id)
logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(logging_dir)
logger.info("Using output dir: %s" % logging_dir)

data_dir = args["data_dir"]
if data_dir != "":
    update_data_dir(data_dir)

num_epochs = args["n_epochs"]
val_per_n_steps = args["val_per_n_steps"]
cropharvest_per_n_validations = args["cropharvest_per_n_validations"]
cropharvest_val_n_per_class = args["cropharvest_val_n_per_class"]
dynamic_world_loss_weight = args["dynamic_world_loss_weight"]
max_learning_rate = args["max_learning_rate"]
min_learning_rate = args["min_learning_rate"]
warmup_epochs = args["warmup_epochs"]
weight_decay = args["weight_decay"]
batch_size = args["batch_size"]

mask_strategies: Tuple[str, ...] = tuple(args["mask_strategies"])
if (len(mask_strategies) == 1) and (mask_strategies[0] == "all"):
    mask_strategies = MASK_STRATEGIES
mask_ratio: float = args["mask_ratio"]

train_url: str = args["train_url"]
val_url: str = args["val_url"]
dataloader_length: int = args["dataloader_length"]

if (batch_size != argparser.get_default("batch_size")) & (
    dataloader_length == argparser.get_default("dataloader_length")
):
    warnings.warn(
        "Dataloader length calculated for a specific batch size. "
        "Set dataloader_length to -1 to recalculate"
    )

skip_finetuning: bool = args["skip_finetuning"]

if path_to_config == "":
    path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

# ------------ Dataloaders -------------------------------------
logger.info("Setting up dataloaders")
mask_params = MaskParams(mask_strategies, mask_ratio)

train_dataset = FranceCropsFullDataset(
    dataset="saget-antoine/francecrops",
    split="train",
    mask_params=mask_params,
    shuffle=True,
    seed=42,
    cache_dir="./cache_train"
)
val_dataset = FranceCropsFullDataset(
    dataset="saget-antoine/francecrops",
    split="validation",
    mask_params=mask_params,
    shuffle=False,
    seed=42,
    cache_dir="./cache_val"
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

if dataloader_length == -1:
    logger.info("Finding train dataloader length")
    dataloader_length = 0
    for _ in train_dataloader:
        dataloader_length += 1
    logger.info("train_dataloader length: ", dataloader_length)

# ------------ Model -----------------------------------------
logger.info("Setting up model")
model = Presto.construct(**model_kwargs)
model.to(device)

# ------------ Model hyperparameters -------------------------------------
param_groups = param_groups_weight_decay(model, weight_decay)
optimizer = optim.AdamW(param_groups, lr=max_learning_rate, betas=(0.9, 0.95))
mse = LossWrapper(nn.MSELoss())
ce = LossWrapper(nn.CrossEntropyLoss())

training_config = {
    "model": model.__class__,
    "encoder": model.encoder.__class__,
    "decoder": model.decoder.__class__,
    "optimizer": optimizer.__class__.__name__,
    "eo_loss": mse.loss.__class__.__name__,
    "dynamic_world_loss": ce.loss.__class__.__name__,
    "device": device,
    **args,
    **model_kwargs,
}

lowest_validation_loss = None
best_val_epoch = 0
training_step = 0
num_validations = 0

with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:
    for epoch in tqdm_epoch:
        # ------------------------ Training ----------------------------------------
        total_train_loss = 0.0
        total_eo_train_loss = 0.0
        total_num_eo_values_masked = 0
        num_updates_being_captured = 0
        train_size = 0
        model.train()
        for epoch_step, b in enumerate(train_dataloader):
            mask, x, y = b["mask"].to(device), b["x"].to(device), b["y"].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            lr = adjust_learning_rate(
                optimizer,
                epoch_step / dataloader_length + epoch,
                warmup_epochs,
                num_epochs,
                max_learning_rate,
                min_learning_rate,
            )

            # Get model outputs and calculate loss
            y_pred = model(
                x, mask=mask
            )

            loss = mse(y_pred[mask], y[mask])

            num_eo_masked = len(y_pred[mask])

            total_loss = loss
            total_loss.backward()
            optimizer.step()

            current_batch_size = len(x)
            total_train_loss += total_loss.item()
            total_eo_train_loss += loss.item() * num_eo_masked
            total_num_eo_values_masked += num_eo_masked
            num_updates_being_captured += 1
            train_size += current_batch_size
            training_step += 1

            # ------------------------ Validation --------------------------------------
            if training_step % val_per_n_steps == 0:
                total_val_loss = 0.0
                total_eo_val_loss = 0.0
                total_val_num_eo_values_masked = 0
                num_val_updates_captured = 0
                val_size = 0
                model.eval()
                with torch.no_grad():
                    for b in tqdm(val_dataloader, desc="Validate"):
                        mask, x, y = (
                            b["mask"].to(device),
                            b["x"].to(device),
                            b["y"].to(device),
                        )
                        # Get model outputs and calculate loss
                        y_pred = model(
                            x, mask=mask
                        )
                        loss = mse(y_pred[mask], y[mask])
                        num_eo_masked = len(y_pred[mask])
                        total_loss = loss
                        current_batch_size = len(x)
                        val_size += current_batch_size
                        total_val_loss += total_loss.item()
                        total_eo_val_loss += loss.item() * num_eo_masked
                        total_val_num_eo_values_masked += num_eo_masked
                        num_val_updates_captured += 1

                # ------------------------ Metrics + Logging -------------------------------
                # train_loss now reflects the value against which we calculate gradients
                train_loss = total_train_loss / num_updates_being_captured
                train_eo_loss = total_eo_train_loss / max(total_num_eo_values_masked, 1)

                val_loss = total_val_loss / num_val_updates_captured
                val_eo_loss = total_eo_val_loss / max(total_val_num_eo_values_masked, 1)

                if "train_size" not in training_config and "val_size" not in training_config:
                    training_config["train_size"] = train_size
                    training_config["val_size"] = val_size

                to_log = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_eo_loss": train_eo_loss,
                    "val_eo_loss": val_eo_loss,
                    "training_step": training_step,
                    "epoch": epoch,
                    "lr": lr,
                }
                tqdm_epoch.set_postfix(loss=val_loss)

                if lowest_validation_loss is None or val_loss < lowest_validation_loss:
                    lowest_validation_loss = val_loss
                    best_val_epoch = epoch

                    model_path = logging_dir / Path("models")
                    model_path.mkdir(exist_ok=True, parents=True)

                    best_model_path = model_path / f"{model_name}{epoch}.pt"
                    logger.info(f"Saving best model to: {best_model_path}")
                    torch.save(model.state_dict(), best_model_path)

                # reset training logging
                total_train_loss = 0.0
                total_eo_train_loss = 0.0
                total_num_eo_values_masked = 0
                num_updates_being_captured = 0
                train_size = 0
                num_validations += 1

                model.train()

logger.info(f"Done training, best model saved to {best_model_path}")