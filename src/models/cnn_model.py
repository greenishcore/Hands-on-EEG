"""
CNN model for EEG signal classification.

Architecture mirrors the EEGNet used in the original project notebooks
(``open/app/visialization.ipynb``, ``open/NN/cnn/``), parameterised so
that it can be adapted from the configuration file.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for model training. "
            "Install it from https://pytorch.org/."
        )


class CNNModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """2-D convolutional network for EEG classification.

    The network expects input of shape ``(batch, 1, channels, window_size)``
    and outputs class logits of shape ``(batch, num_classes)``.

    Parameters
    ----------
    num_channels:
        Number of EEG input channels (height of the 2-D input).
    window_size:
        Number of time-point samples (width of the 2-D input).
    num_classes:
        Number of output classes.
    dropout:
        Dropout probability applied after each pooling layer.

    Example
    -------
    >>> import torch
    >>> model = CNNModel(num_channels=32, window_size=128, num_classes=6)
    >>> x = torch.randn(4, 1, 32, 128)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 6])
    """

    def __init__(
        self,
        num_channels: int = 32,
        window_size: int = 128,
        num_classes: int = 6,
        dropout: float = 0.25,
    ) -> None:
        _require_torch()
        super().__init__()
        import torch.nn as nn  # local import so the module is importable without torch

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 4), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 4), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout2 = nn.Dropout(p=dropout)

        # Compute the flattened size dynamically.
        self._flat_size = self._compute_flat_size(num_channels, window_size)

        self.fc1 = nn.Linear(self._flat_size, 128)
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_flat_size(self, num_channels: int, window_size: int) -> int:
        """Return the number of features after the convolutional blocks."""
        import torch

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, window_size)
            out = self._forward_conv(dummy)
        return int(out.numel())

    def _forward_conv(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch.nn.functional as F

        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        return x

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: D102
        import torch.nn.functional as F

        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout3(F.relu(self.fc1(x)))
        return self.fc2(x)

    @classmethod
    def from_config(cls, cfg: object) -> "CNNModel":
        """Instantiate a :class:`CNNModel` from a :class:`~src.config.Config`.

        Parameters
        ----------
        cfg:
            A :class:`~src.config.Config` object.

        Returns
        -------
        CNNModel
        """
        data_cfg = getattr(cfg, "data", None)
        model_cfg = getattr(cfg, "model", None)
        return cls(
            num_channels=getattr(data_cfg, "num_channels", 32) if data_cfg else 32,
            window_size=getattr(data_cfg, "window_size", 128) if data_cfg else 128,
            num_classes=getattr(model_cfg, "num_classes", 6) if model_cfg else 6,
        )

    @classmethod
    def load(
        cls,
        weights_path: str,
        num_channels: int = 32,
        window_size: int = 128,
        num_classes: int = 6,
        map_location: Optional[str] = None,
    ) -> "CNNModel":
        """Load a :class:`CNNModel` from a saved weights file.

        Parameters
        ----------
        weights_path:
            Path to the ``*.pt`` / ``*.pth`` file produced by
            :func:`torch.save`.
        num_channels, window_size, num_classes:
            Architecture parameters – must match the saved model.
        map_location:
            Passed to :func:`torch.load` (e.g. ``"cpu"``).

        Returns
        -------
        CNNModel
            Model in evaluation mode with weights loaded.
        """
        _require_torch()
        import torch

        model = cls(
            num_channels=num_channels,
            window_size=window_size,
            num_classes=num_classes,
        )
        state = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state)
        model.eval()
        return model
