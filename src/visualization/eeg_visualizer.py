"""
EEG visualizer – refactored, portable replacement for ``open/app/polt.py``.

Key improvements over the original script:

* **No hardcoded paths** – data file is supplied at construction time.
* **Object-oriented design** – state is encapsulated in :class:`EEGVisualizer`.
* **Parameterised** – channel count, window size, colours, and other
  settings are configurable via constructor arguments or a
  :class:`~src.config.Config` object.
* **Error handling** – descriptive exceptions on bad input.
* **Cross-platform** – uses :mod:`pathlib` for all file access.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.data.loader import load_eeg_csv
from src.utils.logger import get_logger

_log = get_logger(__name__)


class EEGVisualizer:
    """Animated EEG waveform viewer using VisPy.

    Parameters
    ----------
    file_path:
        Path to the EEG CSV file to visualise (relative or absolute).
    num_channels:
        Number of EEG channels.  Defaults to 32.
    window_size:
        Number of samples shown in the sliding window.  Defaults to 128.
    color:
        Hex colour string for the EEG traces.  Defaults to ``"#00251C"``.
    title:
        Plot title displayed inside the canvas.
    animation_interval:
        Interval between animation frames in seconds.  Defaults to 0.01.
    line_width:
        Width of each channel trace in pixels.  Defaults to 2.
    config:
        Optional :class:`~src.config.Config` instance whose
        ``visualization`` and ``data`` sections supply default values.
        Explicit keyword arguments take precedence over *config* values.

    Example
    -------
    >>> viz = EEGVisualizer("data/sample.csv")
    >>> viz.run()
    """

    def __init__(
        self,
        file_path: Union[str, os.PathLike],
        num_channels: int = 32,
        window_size: int = 128,
        color: str = "#00251C",
        title: str = "32-Channel EEG Data",
        animation_interval: float = 0.01,
        line_width: int = 2,
        config: Optional[object] = None,
    ) -> None:
        # Apply config defaults first, then explicit kwargs override.
        if config is not None:
            vis_cfg = getattr(config, "visualization", None)
            data_cfg = getattr(config, "data", None)
            if vis_cfg is not None:
                color = getattr(vis_cfg, "color", color)
                title = getattr(vis_cfg, "title", title)
                animation_interval = (
                    getattr(vis_cfg, "animation_interval_ms", animation_interval * 1000)
                    / 1000.0
                )
                line_width = getattr(vis_cfg, "line_width", line_width)
            if data_cfg is not None:
                num_channels = getattr(data_cfg, "num_channels", num_channels)
                window_size = getattr(data_cfg, "window_size", window_size)

        self.file_path = Path(file_path).resolve()
        self.num_channels = num_channels
        self.window_size = window_size
        self.color = color
        self.title = title
        self.animation_interval = animation_interval
        self.line_width = line_width

        # Internal state
        self._data: Optional[np.ndarray] = None
        self._start_pos: int = 0
        self._canvas = None
        self._view = None
        self._lines: list = []
        self._timer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "EEGVisualizer":
        """Load EEG data from :attr:`file_path`.

        Returns
        -------
        EEGVisualizer
            *self*, so calls can be chained: ``viz.load().run()``.
        """
        self._data = load_eeg_csv(
            self.file_path, expected_channels=self.num_channels
        )
        return self

    def build(self) -> "EEGVisualizer":
        """Construct the VisPy scene, lines, and animation timer.

        :meth:`load` must be called before :meth:`build`.

        Returns
        -------
        EEGVisualizer
            *self*, for chaining.

        Raises
        ------
        RuntimeError
            If data has not been loaded yet.
        ImportError
            If ``vispy`` is not installed.
        """
        if self._data is None:
            raise RuntimeError(
                "No data loaded. Call EEGVisualizer.load() before build()."
            )

        try:
            import vispy.app
            import vispy.scene
            from vispy.scene import visuals
        except ImportError as exc:
            raise ImportError(
                "The 'vispy' package is required for EEG visualisation. "
                "Install it with: pip install vispy"
            ) from exc

        data = self._data  # shape: (channels, time_points)

        canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
        view = canvas.central_widget.add_view()

        lines = []
        for _ in range(self.num_channels):
            line = visuals.Line(
                pos=np.array([[0, 0], [1, 1]]),
                width=self.line_width,
                color=self.color,
            )
            lines.append(line)
            view.add(line)

        view.camera.rect = (0, -800, data.shape[1], 800)
        view.camera.set_range()

        title_node = vispy.scene.visuals.Text(
            self.title,
            font_size=16,
            anchor_x="center",
            anchor_y="top",
            pos=[data.shape[1] / 2, -850],
        )
        view.add(title_node)

        timer = vispy.app.Timer(
            connect=self._update, interval=self.animation_interval
        )

        self._canvas = canvas
        self._view = view
        self._lines = lines
        self._timer = timer
        self._start_pos = 0

        _log.info("VisPy scene built for %d channels.", self.num_channels)
        return self

    def run(self) -> None:
        """Start the animation and enter the VisPy event loop.

        Calls :meth:`load` and :meth:`build` automatically if they have
        not been called yet.
        """
        import vispy.app

        if self._data is None:
            self.load()
        if self._canvas is None:
            self.build()

        self._timer.start()
        _log.info("Starting EEG visualiser – close the window to exit.")
        vispy.app.run()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update(self, event: object) -> None:  # noqa: ANN001
        """Timer callback – advance the sliding window by one sample."""
        data = self._data
        ws = self.window_size
        start = self._start_pos
        end = start + ws

        if end > data.shape[1]:
            self._start_pos = 0
            return

        time_axis = np.arange(ws) + start

        for i in range(self.num_channels):
            pos = np.stack([time_axis, data[i, start:end]], axis=1)
            self._lines[i].set_data(pos=pos)

        self._start_pos += 1
