"""
Configuration management for the Hands-on-EEG project.

Supports loading settings from a YAML file and environment variables,
providing a single source of truth for all configurable parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class DataConfig:
    """Data-related configuration."""

    data_dir: str = "data"
    """Root directory for EEG data files (relative or absolute)."""

    num_channels: int = 32
    """Expected number of EEG channels."""

    sampling_rate: int = 128
    """Sampling rate of the EEG device in Hz."""

    window_size: int = 128
    """Sliding-window size (number of samples)."""

    window_step: int = 1
    """Step size between consecutive windows."""

    labels: List[str] = field(
        default_factory=lambda: [
            "lefthand",
            "read",
            "rest",
            "walkbase",
            "walkl",
            "walkfocus",
        ]
    )
    """Class labels for EEG tasks."""


@dataclass
class VisualizationConfig:
    """Visualization-related configuration."""

    color: str = "#00251C"
    """Line colour for EEG traces."""

    title: str = "32-Channel EEG Data"
    """Plot title."""

    animation_interval_ms: float = 10.0
    """Timer interval for the sliding-window animation (milliseconds)."""

    line_width: int = 2
    """Width of each channel trace in pixels."""


@dataclass
class ModelConfig:
    """Model training configuration."""

    batch_size: int = 32
    """Training batch size."""

    num_epochs: int = 100
    """Number of training epochs."""

    learning_rate: float = 1e-3
    """Initial learning rate."""

    model_dir: str = "models"
    """Directory where trained model weights are saved."""

    num_classes: int = 6
    """Number of output classes."""


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    """Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

    log_file: Optional[str] = None
    """Path to the log file. If None, only console logging is used."""

    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    """Log record format string."""


@dataclass
class Config:
    """Top-level project configuration.

    Example
    -------
    >>> cfg = load_config("config.yaml")
    >>> print(cfg.data.num_channels)
    32
    """

    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_path(self, relative_path: str) -> Path:
        """Return *relative_path* resolved relative to the project root.

        The project root is determined as the parent of the directory that
        contains this module, i.e. the repository root.

        Parameters
        ----------
        relative_path:
            A path string that may be absolute or relative.

        Returns
        -------
        Path
            An absolute :class:`~pathlib.Path`.
        """
        p = Path(relative_path)
        if p.is_absolute():
            return p
        # Project root = two levels above this file (src/config/config.py)
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / p


def _apply_env_overrides(cfg: Config) -> None:
    """Override config values with environment variables when present.

    Supported environment variables
    --------------------------------
    ``EEG_DATA_DIR``        – overrides ``cfg.data.data_dir``
    ``EEG_NUM_CHANNELS``    – overrides ``cfg.data.num_channels``
    ``EEG_SAMPLING_RATE``   – overrides ``cfg.data.sampling_rate``
    ``EEG_WINDOW_SIZE``     – overrides ``cfg.data.window_size``
    ``EEG_MODEL_DIR``       – overrides ``cfg.model.model_dir``
    ``EEG_LOG_LEVEL``       – overrides ``cfg.logging.level``
    ``EEG_LOG_FILE``        – overrides ``cfg.logging.log_file``
    """
    env_map = {
        "EEG_DATA_DIR": ("data", "data_dir", str),
        "EEG_NUM_CHANNELS": ("data", "num_channels", int),
        "EEG_SAMPLING_RATE": ("data", "sampling_rate", int),
        "EEG_WINDOW_SIZE": ("data", "window_size", int),
        "EEG_MODEL_DIR": ("model", "model_dir", str),
        "EEG_LOG_LEVEL": ("logging", "level", str),
        "EEG_LOG_FILE": ("logging", "log_file", str),
    }
    for env_var, (section, attr, cast) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            setattr(getattr(cfg, section), attr, cast(value))


def load_config(path: Optional[str] = None) -> Config:
    """Load project configuration from a YAML file and environment variables.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.  If *None* or the file does not
        exist, default values are used.

    Returns
    -------
    Config
        A fully initialised :class:`Config` instance.

    Raises
    ------
    ImportError
        If *path* is provided but the ``pyyaml`` package is not installed.
    ValueError
        If the YAML file contains unexpected top-level keys.
    """
    cfg = Config()

    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "The 'pyyaml' package is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                )
            with config_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

            allowed_keys = {"data", "visualization", "model", "logging"}
            unknown = set(raw.keys()) - allowed_keys
            if unknown:
                raise ValueError(
                    f"Unknown top-level config keys: {unknown}. "
                    f"Allowed keys are: {allowed_keys}"
                )

            if "data" in raw:
                for k, v in raw["data"].items():
                    setattr(cfg.data, k, v)
            if "visualization" in raw:
                for k, v in raw["visualization"].items():
                    setattr(cfg.visualization, k, v)
            if "model" in raw:
                for k, v in raw["model"].items():
                    setattr(cfg.model, k, v)
            if "logging" in raw:
                for k, v in raw["logging"].items():
                    setattr(cfg.logging, k, v)

    _apply_env_overrides(cfg)
    return cfg
