# sometimes, runs fail
# This (hacky) script lets
# eval results be logged to the
# right wandb run
import argparse
import json
import logging
from pathlib import Path
from typing import List, cast

import torch
from tqdm import tqdm

from presto.eval import (
    AlgaeBloomsEval,
    CropHarvestEval,
    CroptypeFranceEval,
    EuroSatEval,
    EvalTask,
    FuelMoistureEval,
)
from presto.presto import Presto
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    default_model_path,
    device,
    initialize_logging,
    seed_everything,
    timestamp_dirname,
    update_data_dir,
)

seed_everything()
logger = logging.getLogger("__main__")

argparser = argparse.ArgumentParser()
argparser.add_argument("--path_to_state_dict", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Data is stored in <data_dir>/data. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Output is stored in <data_dir>/output. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument("--fully_supervised", dest="fully_supervised", action="store_true")
argparser.set_defaults(fully_supervised=False)
args = argparser.parse_args().__dict__

path_to_state_dict = args["path_to_state_dict"]
path_to_config = args["path_to_config"]
fully_supervised = args["fully_supervised"]
data_dir = args["data_dir"]
if data_dir != "":
    update_data_dir(data_dir)

output_parent_dir = Path(args["output_dir"]) if args["output_dir"] else Path(__file__).parent
run_id = None

logging_dir = output_parent_dir / "output" / timestamp_dirname(run_id)
logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(logging_dir)
logger.info("Using output dir: %s" % logging_dir)

if path_to_config == "":
    path_to_config = config_dir / "default.json"
logger.info("Loading config from %s" % path_to_config)
model_kwargs = json.load(Path(path_to_config).open("r"))
model = Presto.construct(**model_kwargs)

if not fully_supervised:
    if path_to_state_dict == "":
        path_to_state_dict = default_model_path
    logger.info("Loading params from %s" % path_to_state_dict)
    model.load_state_dict(torch.load(path_to_state_dict, map_location=device))
model.to(device)

logger.info("Loading evaluation tasks")
seeds = [0, DEFAULT_SEED, 84]
eval_task_list: List[EvalTask] = [
    *[
        CropHarvestEval(country="Brazil", ignore_dynamic_world=idw, seed=seed)
        for idw in [True, False]
        for seed in seeds
    ],
]

result_dict = {}
for eval_task in tqdm(eval_task_list, desc="Full Evaluation"):
    model_modes = ["finetune", "Regression", "Random Forest"]
    logger.info(eval_task.name)

    results = eval_task.finetuning_results(model, model_modes=model_modes)
    result_dict.update(results)
    logger.info(json.dumps(results, indent=2))

    eval_task.clear_data()

eval_results_file = logging_dir / "results.json"
logger.info("Saving eval results to file %s" % eval_results_file)
with open(eval_results_file, "w") as f:
    json.dump(result_dict, f)
