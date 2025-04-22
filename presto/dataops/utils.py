from typing import List, Optional
import warnings

import numpy as np
import torch

from .pipelines.dynamicworld import DynamicWorld2020_2021
from .pipelines.s1_s2_era5_srtm import (
    BANDS,
    NORMED_BANDS,
    REMOVED_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
    ADD_BY,
    DIVIDE_BY,
)

def calculate_ndvi(input_array, s2_bands):
        r"""
        Given an input array of shape [timestep, bands] or [batches, timesteps, shapes]
        where bands == len(bands), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)
        """
        band_1, band_2 = "B8", "B4"

        num_dims = len(input_array.shape)
        if num_dims == 2:
            band_1_np = input_array[:, s2_bands.index(band_1)]
            band_2_np = input_array[:, s2_bands.index(band_2)]
        elif num_dims == 3:
            band_1_np = input_array[:, :, s2_bands.index(band_1)]
            band_2_np = input_array[:, :, s2_bands.index(band_2)]
        else:
            raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            if isinstance(band_1_np, np.ndarray):
                return np.where(
                    (band_1_np + band_2_np) > 0,
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    0,
                )
            else:
                return torch.where(
                    (band_1_np + band_2_np) > 0,
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    0,
                )

def construct_single_presto_input(
    s2: Optional[torch.Tensor] = None,
    s2_bands: Optional[List[str]] = None,
    normalize: bool = True,
    ndvi: bool = True
):
    """
    Inputs are paired into a tensor input <X> and a list <X>_bands, which describes <X>.

    <X> should have shape (num_timesteps, len(<X>_bands)), with the following bands possible for
    each input:

    s2: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

    dynamic_world is a 1d input of shape (num_timesteps,) representing the dynamic world classes
        of each timestep for that pixel
    """
    num_timesteps_list = [x.shape[0] for x in [s2] if x is not None]

    assert len(num_timesteps_list) > 0
    assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
    num_timesteps = num_timesteps_list[0]
    ndvi_len = 1 if ndvi else 0
    mask, x = torch.ones(num_timesteps, len(s2_bands) + ndvi_len), torch.zeros(num_timesteps, len(s2_bands) + ndvi_len)

    for band_group in [
        (s2, s2_bands, S2_BANDS),
    ]:
        data, input_bands, output_bands = band_group
        if data is not None:
            assert input_bands is not None
        else:
            continue

        # construct a mapping from the input bands to the expected bands
        kept_input_band_idxs = [i for i, val in enumerate(input_bands) if val in output_bands]
        kept_input_band_names = [val for val in input_bands if val in output_bands]

        input_to_output_mapping = [s2_bands.index(val) for val in kept_input_band_names]

        x[:, input_to_output_mapping] = data[:, kept_input_band_idxs]
        mask[:, input_to_output_mapping] = 0

    if normalize:
        if isinstance(x, np.ndarray):
            x = ((x + ADD_BY) / DIVIDE_BY).astype(np.float32)
        else:
            x = (x + torch.tensor(ADD_BY)) / torch.tensor(DIVIDE_BY)
    if ndvi:        
        if len(x.shape) == 2:
            x[:, len(s2_bands)] = calculate_ndvi(x, s2_bands)
        else:
            x[:, :, len(s2_bands)] = calculate_ndvi(x, s2_bands)
        mask[:, len(s2_bands)] = 0
    return x, mask