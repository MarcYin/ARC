import numpy as np
from arc.robust_smoothing import robust_smooth

from typing import Tuple

def calculate_ndvi(s2_refs: np.array) -> np.array:
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) given an array of satellite image reflectances.

    :param s2_refs: An array of satellite image reflectances.
    :return: An array of NDVI values.
    """
    return (s2_refs[:, 7] - s2_refs[:, 2]) / (s2_refs[:, 7] + s2_refs[:, 2])


def time_series_filter(array, udoys):
    mask = array > 0

    # Zero out non-vegetation pixels
    array[~mask] = 0

    # Smooth the array
    sarray, w = robust_smooth(array=array*1., Warray=mask*1., x=udoys, s=1, d=1, iterations=2, axis=0)
    # diff = array - sarray
    # diff[~mask] = np.nan

    # diff_thresh = np.nanpercentile(abs(diff), 90, axis=0)
    # time_series_mask = (abs(diff) < diff_thresh[None])
    time_series_mask = w > 0.5
    return time_series_mask


def ndvi_filter(s2_refs: np.array, s2_uncs: np.array, doys: np.array, s2_angles: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Filters and smooths NDVI values, removing non-vegetation pixels and smoothing over time.

    :param s2_refs: An array of satellite image reflectances.
    :param s2_uncs: An array of associated uncertainties.
    :param doys: An array of day of year values.
    :return: Transposed s2_refs, s2_uncs, doys and s2_angles arrays.
    """
    # Calculate NDVI
    ndvi = calculate_ndvi(s2_refs)

    # Ensure unique days of the year and corresponding indices
    udoys, inds = np.unique(doys, return_index=True)

    # Transpose references and uncertainties based on unique indices
    s2_refs = s2_refs[inds].transpose(1, 0, 2, 3)
    s2_uncs = s2_uncs[inds].transpose(1, 0, 2, 3)

    # Adjust doys array based on unique indices
    doys = doys[inds]

    s2_angles = s2_angles[:, inds]

    # Scale udoys to roughly have step of 1
    udoys = udoys / np.diff(udoys).mean()

    # Filter NDVI values and reshape to 1D
    array = ndvi[inds].reshape(len(udoys), -1)
    time_series_mask1 = time_series_filter(array, udoys)

    # Filter B8A values and reshape to 1D
    array = s2_refs[7].reshape(len(udoys), -1)
    time_series_mask2 = time_series_filter(array, udoys)

    array = s2_refs[2].reshape(len(udoys), -1)
    time_series_mask3 = time_series_filter(array, udoys)
    time_series_mask = time_series_mask1 & time_series_mask2 & time_series_mask3

    time_series_mask = time_series_mask.reshape(s2_refs.shape[1], s2_refs.shape[2], s2_refs.shape[3])

    # Apply the mask to the references and uncertainties
    s2_refs[:, ~time_series_mask] = np.nan
    s2_uncs[:, ~time_series_mask] = np.nan

    return s2_refs.transpose(1, 0, 2, 3), s2_uncs.transpose(1, 0, 2, 3), doys, s2_angles


def save_data(file_path, post_bio_tensor, post_bio_unc_tensor, dat, geotransform, crs, mask, doys):
    """
    Save data to a npz file.

    Args:
        file_path (str): The path of the npz file.
        post_bio_tensor (np.array): The post bio tensor.
        post_bio_unc_tensor (np.array): The post bio unc tensor.
        dat (np.array): The data array.
        geotransform (np.array): The geotransform.
        crs (str): The CRS.
        mask (np.array): The mask.
    """
    np.savez(file_path, post_bio_tensor=post_bio_tensor, post_bio_unc_tensor=post_bio_unc_tensor, dat=dat, geotransform=geotransform, crs=crs, mask=mask, doys=doys)
