from collections import namedtuple
from dataclasses import dataclass
from random import choice, randint, random, sample
from typing import Any, List, Tuple

import numpy as np
from pandas.compat._optional import import_optional_dependency

from .pipelines.dynamicworld import DynamicWorld2020_2021
from .pipelines.s1_s2_era5_srtm import (
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    NUM_TIMESTEPS,
    TIMESTEPS_IDX,
)

MASK_STRATEGIES = (
    "group_bands",
    "random_timesteps",
    "chunk_timesteps",
    "random_combinations",
)

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
BAND_EXPANSION = [len(x) for x in BANDS_GROUPS_IDX.values()]

MaskedExample = namedtuple(
    "MaskedExample",
    ["mask_eo", "x_eo", "y_eo", "strategy"],
)


def make_mask(strategy: str, mask_ratio: float) -> Tuple[np.ndarray]:
    """
    Make a mask for a given strategy and percentage of masked values.
    Args:
        strategy: The masking strategy to use. One of MASK_STRATEGIES
        mask_ratio: The percentage of values to mask. Between 0 and 1.
    """

    # SRTM is included here, but ignored by Presto
    mask = np.full((NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)), False)
    print(f"mask shape: {mask.shape}")
    num_tokens_to_mask = int((NUM_TIMESTEPS * len(BANDS_GROUPS_IDX)) * mask_ratio)
    print(f"num_tokens_to_mask not multiplied by mask_ratio: {int(NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))}")
    print(f"num_tokens_to_mask: {num_tokens_to_mask}")

    def mask_topography(num_tokens_to_mask, mask_ratio):
        should_flip = random() < mask_ratio
        if should_flip:
            num_tokens_to_mask -= 1
        return num_tokens_to_mask

    def random_masking(mask, num_tokens_to_mask: int):
        if num_tokens_to_mask > 0:
            # then, we flatten the mask
            all_tokens_mask = mask.flatten()
            unmasked_tokens = all_tokens_mask == False
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask]
            all_tokens_mask[idx] = True
            print("all_tokens_mask shape: ", all_tokens_mask.shape)
            print("NUM_TIMESTEPS: ", NUM_TIMESTEPS)
            print("all_tokens_mask[NUM_TIMESTEPS: ] shape: ", all_tokens_mask[NUM_TIMESTEPS:].shape)
            mask = all_tokens_mask[NUM_TIMESTEPS:].reshape((NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        return mask

    # RANDOM BANDS
    if strategy == "random_combinations":
        num_tokens_to_mask = mask_topography(num_tokens_to_mask, mask_ratio)
        mask = random_masking(mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        num_tokens_to_mask = mask_topography(num_tokens_to_mask, mask_ratio)
        # next, we figure out how many tokens we can mask
        num_band_groups_to_mask = int(num_tokens_to_mask / NUM_TIMESTEPS)
        num_tokens_to_mask -= NUM_TIMESTEPS * num_band_groups_to_mask
        assert num_tokens_to_mask >= 0
        # tuple because of mypy, which thinks lists can only hold one type
        band_groups: List[Any] = list(range(len(BANDS_GROUPS_IDX)))
        band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        for band_group in band_groups_to_mask:
            mask[:, band_group] = True
        mask = random_masking(mask, num_tokens_to_mask)

    # RANDOM TIMESTEPS
    elif strategy == "random_timesteps":
        num_tokens_to_mask = mask_topography(num_tokens_to_mask, mask_ratio)
        # +1 for dynamic world, -1 for the SRTM
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX)))
        num_tokens_to_mask -= (len(BANDS_GROUPS_IDX)) * timesteps_to_mask
        timesteps = sample(TIMESTEPS_IDX, k=timesteps_to_mask)
        mask[timesteps] = True
        mask = random_masking(mask, num_tokens_to_mask)
    elif strategy == "chunk_timesteps":
        num_tokens_to_mask = mask_topography(num_tokens_to_mask, mask_ratio)
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX)))
        num_tokens_to_mask -= (len(BANDS_GROUPS_IDX)) * timesteps_to_mask
        start_idx = randint(0, NUM_TIMESTEPS - timesteps_to_mask)
        mask[start_idx : start_idx + timesteps_to_mask] = True  # noqa
        mask = random_masking(mask, num_tokens_to_mask)
    else:
        raise ValueError(f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    return np.repeat(mask, BAND_EXPANSION, axis=1)


@dataclass
class MaskParams:
    strategies: Tuple[str, ...] = ("NDVI",)
    ratio: float = 0.5

    def __post_init__(self):
        for strategy in self.strategies:
            assert strategy in [
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ]

    def mask_data(self, eo_data: np.ndarray):
        strategy = choice(self.strategies)
        mask = make_mask(strategy=strategy, mask_ratio=self.ratio)
        x = eo_data * ~mask
        y = np.zeros(eo_data.shape).astype(np.float32)
        y[mask] = eo_data[mask]

        return mask, x, y, strategy


def plot_masked_bands(y_true: np.ndarray, y_pred: np.ndarray, mask_strategy: str):
    """Plot only the masked bands over time"""
    ncols = len(BANDS_GROUPS_IDX[mask_strategy])
    plt = import_optional_dependency("matplotlib.pyplot")
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 10))
    for i, masked_band_idx in enumerate(BANDS_GROUPS_IDX[mask_strategy]):
        if ncols == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(y_true[:, masked_band_idx], label=f"Actual {mask_strategy} band {i}")
        ax.plot(y_pred[:, masked_band_idx], label=f"Prediced {mask_strategy} band {i}")
        ax.set_title(f"{mask_strategy} band {i}")
        ax.set_ylabel(f"{mask_strategy} band {i}")
        ax.set_xlabel("Time interval")
        ax.legend()
    return fig


def plot_masked_general(example: MaskedExample, y_pred: np.ndarray):
    """Plot all bands over time"""
    plt = import_optional_dependency("matplotlib.pyplot")
    fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(20, 30))

    # Reconstruct eo data
    eo_data_actual = example.x_eo.copy()
    eo_data_actual[example.mask_eo == 1] = example.y_eo[example.mask_eo == 1]
    eo_data_predicted = y_pred

    row_idx = 0
    for band_group, band_indexes in BANDS_GROUPS_IDX.items():
        if row_idx > 6:
            row_idx = 6
        else:
            col_idx = 0
        for b in band_indexes:
            ax = axes[row_idx, col_idx]
            (pred_line,) = ax.plot(eo_data_predicted[:, b], color="orange")
            (actual_line,) = ax.plot(eo_data_actual[:, b], color="blue")
            ax.set_title(NORMED_BANDS[b])
            ax.set_ylabel(band_group)
            col_idx += 1
        row_idx += 1

    fig.legend([pred_line, actual_line], ["Predicted", "Actual"], loc="upper left")
    return fig


def plot_masked(example: MaskedExample, eo_pred: np.ndarray):
    if example.strategy in list(BANDS_GROUPS_IDX.keys()):
        fig = plot_masked_bands(example.y_eo, eo_pred, example.strategy)
    else:
        fig = plot_masked_general(example, eo_pred)
    plt = import_optional_dependency("matplotlib.pyplot")
    plt.suptitle(
        f"Strategy: {example.strategy}",
        size=24,
    )
    fig.subplots_adjust(top=0.15)
    fig.tight_layout()
    return fig
