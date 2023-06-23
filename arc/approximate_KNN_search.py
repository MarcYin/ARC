import numpy as np
import pynndescent
from numba import njit


@njit(fastmath=True)
def non_negative_weighted_euclidean(x, y, weights):
    """
    Computes the non-negative, weighted Euclidean distance between two vectors.
    
    Args:
        x, y : numpy arrays of same length, representing two vectors.
        weights : numpy array of same length as x and y, representing the weight of each coordinate.
    
    Returns:
        The non-negative, weighted Euclidean distance between x and y.
        
    Notes:
        The Euclidean distance is defined as: sqrt(sum((x_i - y_i)^2 * w_i))
        If either corresponding elements of x or y is negative, their contribution to the distance is ignored.
    """
    result = 0.0
    for i in range(x.shape[0]):
        if (x[i] >= 0) and (y[i] >= 0):
            result += ((x[i] - y[i]) ** 2) * weights[i]
    return np.sqrt(result)


def partition_doy(doys, steps=6, overlap=8):
    """
    Partition Day of Year (DOY) data.

    Args:
        doys: A numpy array representing DOY data.
        steps: Number of partitions.
        overlap: Number of overlapping elements between each partition.

    Returns:
        A list of boolean numpy arrays, each representing a partition.
    """
    nodes = np.linspace(doys[0], doys[-1], steps + 1)
    return [(doys >= start - overlap) & (doys <= end + overlap) for start, end in zip(nodes[:-1], nodes[1:])]


def partition_data(parts, data):
    """
    Partition data and calculate the median of each partition.

    Args:
        parts: A list of boolean numpy arrays from 'partition_doy'.
        data: The numpy array to be partitioned.

    Returns:
        A concatenated numpy array of partition medians.
    """
    return np.concatenate(
        [np.nanmedian(data[:, part], axis=1) for part in parts], axis=0
    )


def get_neighbours(s2_refs, s2_errs, arc_refs, doys, steps=10, k=300):
    """
    Calculate nearest neighbors.

    Args:
        s2_refs: The reference observations.
        s2_errs: The error in band values.
        arc_refs: S2 reflectance observations.
        doys: A numpy array representing DOY data.
        steps: Number of partitions.
        k: Number of nearest neighbors to return.

    Returns:
        Nearest neighbor indices.
    """
    parts = partition_doy(doys, steps=steps)

    mean_band_errs = np.repeat(np.nanmean(s2_errs, axis=(1, 2)), len(parts))

    ensemble_parts = partition_data(parts, arc_refs)
    obs_parts = partition_data(parts, s2_refs)
    obs_parts[~np.isfinite(obs_parts)] = -9999

    index = pynndescent.NNDescent(
        ensemble_parts.T,
        metric=non_negative_weighted_euclidean,
        metric_kwds={'weights': 1 / mean_band_errs**2}
    )
    
    return index.query(obs_parts.T, k=k)[0]
