"""
EEG data pre-processing utilities.

Provides functions for common EEG pre-processing steps such as sliding-window
segmentation and per-channel normalisation.
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger
from src.utils.validators import validate_eeg_array

_log = get_logger(__name__)


def sliding_window(
    data: np.ndarray,
    window_size: int = 128,
    step: int = 1,
) -> np.ndarray:
    """Segment an EEG signal into overlapping windows.

    Parameters
    ----------
    data:
        2-D NumPy array of shape ``(channels, time_points)``.
    window_size:
        Number of time-point samples in each window.
    step:
        Number of samples to advance between consecutive windows.

    Returns
    -------
    np.ndarray
        3-D array of shape ``(channels, window_size, num_windows)``
        where ``num_windows = (time_points - window_size) // step + 1``.

    Raises
    ------
    ValueError
        If *data* does not satisfy the 2-D channel × time-point requirement,
        or if *window_size* is larger than the number of time points.
    """
    validate_eeg_array(data, expected_channels=None, min_samples=window_size)

    num_channels, num_samples = data.shape
    num_windows = (num_samples - window_size) // step + 1

    if num_windows <= 0:
        raise ValueError(
            f"window_size ({window_size}) is larger than the number of "
            f"time points ({num_samples})."
        )

    # Build the output array without Python loops using stride tricks.
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=1)
    # sliding_window_view returns shape (channels, num_windows, window_size);
    # subsample along the window axis and transpose to the required layout.
    windows = windows[:, ::step, :]  # (channels, num_windows, window_size)
    windows = windows.transpose(0, 2, 1)  # (channels, window_size, num_windows)

    _log.debug(
        "Sliding window: %d channels × %d samples → %d windows (size=%d, step=%d).",
        num_channels,
        num_samples,
        windows.shape[2],
        window_size,
        step,
    )
    return windows


def normalize_signal(
    data: np.ndarray,
    method: str = "zscore",
    per_channel: bool = True,
) -> np.ndarray:
    """Normalise an EEG signal array.

    Parameters
    ----------
    data:
        2-D NumPy array of shape ``(channels, time_points)``.
    method:
        Normalisation method – one of:

        * ``"zscore"``  – zero mean, unit variance.
        * ``"minmax"``  – scale to the [0, 1] range.
    per_channel:
        When *True* (default) statistics are computed per channel.
        When *False* statistics are computed across the whole array.

    Returns
    -------
    np.ndarray
        Normalised array with the same shape as *data*.

    Raises
    ------
    ValueError
        If an unknown *method* is supplied.
    """
    validate_eeg_array(data, expected_channels=None)

    axis = 1 if per_channel else None

    if method == "zscore":
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        # Avoid division by zero for flat channels.
        std = np.where(std == 0, 1.0, std)
        normalized = (data - mean) / std
    elif method == "minmax":
        data_min = data.min(axis=axis, keepdims=True)
        data_max = data.max(axis=axis, keepdims=True)
        denom = data_max - data_min
        denom = np.where(denom == 0, 1.0, denom)
        normalized = (data - data_min) / denom
    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose 'zscore' or 'minmax'."
        )

    _log.debug("Signal normalised using method='%s', per_channel=%s.", method, per_channel)
    return normalized
