
import numpy as np
import xarray as xr
from typing import Tuple, Optional


def build_ocean_mask_bitmask(
    ds: xr.Dataset,
    mode: str = "first",
    ice_threshold: float = 0.0
) -> np.ndarray:
    """
    Build ocean mask from bitmask field and sea ice fraction
    
    Args:
        ds: xarray Dataset containing 'mask' field
        mode: 'first' uses first timestep, 'majority' uses majority vote across time
        ice_threshold: Maximum sea ice fraction to include (0.0 = no ice)
    
    Returns:
        np.ndarray: Boolean ocean mask (True = valid ocean pixel)
    """
    m = ds["mask"].astype("int16")
    is_water_t = (m & 1) != 0
    is_ice_t = (m & 8) != 0
    
    if mode == "first":
        is_water = is_water_t.isel(time=0)
        is_ice = is_ice_t.isel(time=0)
    elif mode == "majority":
        is_water = (is_water_t.mean("time") > 0.5)
        is_ice = (is_ice_t.mean("time") > 0.5)
    else:
        raise ValueError("mode must be 'first' or 'majority'")
    
    ocean_mask = is_water & (~is_ice)
    
    # Additional filtering using sea ice fraction if available
    if ("sea_ice_fraction" in ds) and (ice_threshold is not None):
        sif = ds["sea_ice_fraction"]
        # Normalize to [0,1] if needed
        if float(sif.max().compute()) > 1.01:
            sif = sif / 100.0
        sif2d = sif.isel(time=0) if mode == "first" else sif.mean("time")
        ocean_mask = ocean_mask & (sif2d <= ice_threshold)
    
    return ocean_mask.values


def normalize_only_ocean(
    da: xr.DataArray,
    ocean_mask: np.ndarray
) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """
    Normalize SST data using mean and std computed only over ocean pixels
    
    Args:
        da: xarray DataArray with SST data (time, lat, lon)
        ocean_mask: Boolean mask indicating valid ocean pixels
    
    Returns:
        Tuple of (normalized_data, mean, std)
    """
    mu = da.where(ocean_mask).mean("time").values.astype(np.float32)
    sigma = da.where(ocean_mask).std("time").values.astype(np.float32)
    
    # Avoid division by zero
    sigma = np.where((sigma < 1e-6) | ~np.isfinite(sigma), 1.0, sigma)
    
    arr = da.values
    arr_n = (arr - mu) / sigma
    arr_n[:, ~ocean_mask] = np.nan
    
    return xr.DataArray(arr_n, coords=da.coords, dims=da.dims), mu, sigma


def load_and_preprocess_sst(
    nc_path: str,
    train_slice: slice,
    val_slice: slice,
    test_slice: Optional[slice] = None,
    mask_mode: str = "first",
    ice_threshold: float = 0.0
) -> dict:
    """
    Load and preprocess SST data from netCDF file
    
    Args:
        nc_path: Path to netCDF file
        train_slice: Time slice for training data
        val_slice: Time slice for validation data
        test_slice: Time slice for test data (optional)
        mask_mode: Method for building ocean mask
        ice_threshold: Sea ice threshold for masking
    
    Returns:
        Dictionary containing preprocessed data and metadata
    """
    ds = xr.open_dataset(nc_path, decode_cf=True)
    sst = ds["sst"]

    ocean_mask = build_ocean_mask_bitmask(ds, mode=mask_mode, ice_threshold=ice_threshold)
    H, W = int(ocean_mask.shape[0]), int(ocean_mask.shape[1])

    sst_train = sst.sel(time=train_slice)
    sst_val = sst.sel(time=val_slice)
    
    # Normalize using training statistics
    sst_train_norm, mu, sigma = normalize_only_ocean(sst_train, ocean_mask)
    
    # Apply same normalization to validation
    arr_val = (sst_val.values - mu) / sigma
    arr_val[:, ~ocean_mask] = np.nan
    sst_val_norm = xr.DataArray(arr_val, coords=sst_val.coords, dims=sst_val.dims)
    
    result = {
        'sst_train_norm': sst_train_norm,
        'sst_val_norm': sst_val_norm,
        'ocean_mask': ocean_mask,
        'mean': mu,
        'std': sigma,
        'height': H,
        'width': W,
        'train_shape': sst_train_norm.shape,
        'val_shape': sst_val_norm.shape
    }
    
    if test_slice is not None:
        sst_test = sst.sel(time=test_slice)
        arr_test = (sst_test.values - mu) / sigma
        arr_test[:, ~ocean_mask] = np.nan
        sst_test_norm = xr.DataArray(arr_test, coords=sst_test.coords, dims=sst_test.dims)
        result['sst_test_norm'] = sst_test_norm
        result['test_shape'] = sst_test_norm.shape
    
    return result


def denormalize_predictions(
    predictions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ocean_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Denormalize predictions back to original scale
    
    Args:
        predictions: Normalized predictions
        mean: Mean used for normalization
        std: Std used for normalization
        ocean_mask: Optional ocean mask to apply
    
    Returns:
        Denormalized predictions
    """
    denorm = predictions * std + mean
    
    if ocean_mask is not None:
        denorm[..., ~ocean_mask] = np.nan
    
    return denorm

