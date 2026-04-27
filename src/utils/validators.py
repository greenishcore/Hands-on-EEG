"""
Data validation utilities for the Hands-on-EEG project.

Provides functions to assert that loaded data conforms to the expected
EEG format before processing begins.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

_log = get_logger(__name__)


def validate_eeg_dataframe(
    df: pd.DataFrame,
    expected_channels: int = 32,
    allow_header_row: bool = True,
) -> pd.DataFrame:
    """Validate a raw EEG :class:`~pandas.DataFrame` loaded from CSV.

    The EMOTIV Plex CSV format stores channel data in columns 1..N and
    contains a header row at index 0 (row of channel names).  After
    stripping the header row and the first column (timestamp / counter),
    the remaining data should have *expected_channels* columns.

    Parameters
    ----------
    df:
        DataFrame loaded from a CSV file (``header=None``).
    expected_channels:
        Number of EEG channels to expect.
    allow_header_row:
        When *True*, row 0 is treated as a header and skipped before
        channel-count validation.

    Returns
    -------
    pd.DataFrame
        The original *df* (not modified), returned for convenience.

    Raises
    ------
    ValueError
        If the DataFrame does not contain the expected number of channels.
    """
    if df.empty:
        raise ValueError("DataFrame is empty – check that the CSV file contains data.")

    data_slice = df.iloc[1:, 1:] if allow_header_row else df.iloc[:, 1:]

    actual_channels = data_slice.shape[1]
    if actual_channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} EEG channels but found {actual_channels}. "
            "Check that the CSV file has the correct format."
        )

    _log.debug(
        "DataFrame validated: %d channels, %d time points.",
        actual_channels,
        data_slice.shape[0],
    )
    return df


def validate_eeg_array(
    data: np.ndarray,
    expected_channels: Optional[int] = 32,
    min_samples: int = 1,
) -> np.ndarray:
    """Validate a NumPy EEG array with shape ``(channels, time_points)``.

    Parameters
    ----------
    data:
        2-D NumPy array of shape ``(channels, time_points)``.
    expected_channels:
        If provided, asserts that ``data.shape[0] == expected_channels``.
        Pass *None* to skip the channel-count check.
    min_samples:
        Minimum number of time-point samples required.

    Returns
    -------
    np.ndarray
        The original *data* array (not modified), returned for convenience.

    Raises
    ------
    ValueError
        If the array does not satisfy the validation criteria.
    """
    if data.ndim != 2:
        raise ValueError(
            f"EEG array must be 2-D (channels × time_points), got shape {data.shape}."
        )

    if expected_channels is not None and data.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels (rows) but got {data.shape[0]}."
        )

    if data.shape[1] < min_samples:
        raise ValueError(
            f"Array has only {data.shape[1]} time-point(s); at least {min_samples} required."
        )

    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(
            f"EEG array must contain floating-point values, got dtype={data.dtype}."
        )

    _log.debug(
        "Array validated: %d channels × %d samples.", data.shape[0], data.shape[1]
    )
    return data
