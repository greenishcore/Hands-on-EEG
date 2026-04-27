"""
EEG data loading utilities.

Provides portable, cross-platform functions to load EEG data from CSV
files produced by the EMOTIV Plex EEG device.  All path handling is
done via :class:`pathlib.Path`, so the same code runs on Windows, macOS,
and Linux.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validators import validate_eeg_dataframe

_log = get_logger(__name__)


def load_eeg_csv(
    file_path: Union[str, os.PathLike],
    expected_channels: int = 32,
    skip_header_row: bool = True,
    skip_first_col: bool = True,
) -> np.ndarray:
    """Load a single EEG CSV file and return a NumPy array.

    The EMOTIV Plex CSV format stores data with a channel-name header row
    and a counter/timestamp first column.  By default both are skipped so
    that the returned array contains only numeric EEG signal values.

    Parameters
    ----------
    file_path:
        Path to the CSV file (relative or absolute).  Relative paths are
        resolved from the current working directory.
    expected_channels:
        Number of EEG channels to validate against.  Set to *None* to skip
        channel-count validation.
    skip_header_row:
        When *True* (default) the first row of the file is treated as a
        header and skipped.
    skip_first_col:
        When *True* (default) the first column (counter / timestamp) is
        dropped.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(channels, time_points)`` with ``float64`` values.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the data does not conform to the expected format.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"EEG data file not found: {path}")

    _log.info("Loading EEG data from: %s", path)

    try:
        df = pd.read_csv(path, header=None)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc

    if expected_channels is not None:
        validate_eeg_dataframe(
            df,
            expected_channels=expected_channels,
            allow_header_row=skip_header_row,
        )

    # Slice out numeric data
    row_start = 1 if skip_header_row else 0
    col_start = 1 if skip_first_col else 0
    data_slice = df.iloc[row_start:, col_start:]

    try:
        data = data_slice.astype(float).values.T  # shape: (channels, time_points)
    except ValueError as exc:
        raise ValueError(
            f"EEG data contains non-numeric values in '{path}': {exc}"
        ) from exc

    _log.info(
        "Loaded: %d channels × %d time points.", data.shape[0], data.shape[1]
    )
    return data


def load_multiple_csv(
    file_paths: List[Union[str, os.PathLike]],
    expected_channels: int = 32,
    skip_header_row: bool = True,
    skip_first_col: bool = True,
) -> List[np.ndarray]:
    """Load multiple EEG CSV files and return a list of NumPy arrays.

    Parameters
    ----------
    file_paths:
        List of paths to CSV files.
    expected_channels:
        Number of EEG channels each file must contain.
    skip_header_row:
        See :func:`load_eeg_csv`.
    skip_first_col:
        See :func:`load_eeg_csv`.

    Returns
    -------
    list of np.ndarray
        One array per file, each of shape ``(channels, time_points)``.
    """
    results: List[np.ndarray] = []
    for fp in file_paths:
        data = load_eeg_csv(
            fp,
            expected_channels=expected_channels,
            skip_header_row=skip_header_row,
            skip_first_col=skip_first_col,
        )
        results.append(data)
    _log.info("Loaded %d EEG file(s).", len(results))
    return results


def find_eeg_files(
    directory: Union[str, os.PathLike],
    pattern: str = "*.csv",
    recursive: bool = True,
) -> List[Path]:
    """Search a directory tree for EEG CSV files.

    Parameters
    ----------
    directory:
        Root directory to search.
    pattern:
        Glob pattern for file names (default: ``"*.csv"``).
    recursive:
        When *True* (default) the search descends into subdirectories.

    Returns
    -------
    list of Path
        Sorted list of matching :class:`~pathlib.Path` objects.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    """
    root = Path(directory).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    glob_fn = root.rglob if recursive else root.glob
    files = sorted(glob_fn(pattern))
    _log.info("Found %d file(s) matching '%s' in %s.", len(files), pattern, root)
    return files
