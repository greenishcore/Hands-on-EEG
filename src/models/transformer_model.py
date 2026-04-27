"""
Transformer model for EEG signal classification.

Architecture is inspired by the Transformer notebooks in
``open/NN/transformer/`` and ``open/NN/transformer_old/``.  The model
treats each time step as a token and applies multi-head self-attention
over the sequence.
"""

from __future__ import annotations

import math
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


class _PositionalEncoding(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        _require_torch()
        super().__init__()
        import torch
        import torch.nn as nn

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add positional encoding to *x* of shape ``(batch, seq_len, d_model)``."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Transformer encoder for EEG classification.

    The input tensor is expected to have shape
    ``(batch, channels, window_size)``.  Each channel time-series is
    projected to an embedding dimension and then encoded by a stack of
    Transformer encoder layers before final classification.

    Parameters
    ----------
    num_channels:
        Number of EEG input channels (used as the sequence length).
    window_size:
        Number of time-point samples (input feature dimension per token).
    num_classes:
        Number of output classes.
    d_model:
        Transformer embedding dimension.
    nhead:
        Number of multi-head attention heads.  Must divide *d_model*.
    num_layers:
        Number of Transformer encoder layers.
    dim_feedforward:
        Hidden size of the feed-forward sub-layer.
    dropout:
        Dropout probability.

    Example
    -------
    >>> import torch
    >>> model = TransformerModel(num_channels=32, window_size=128, num_classes=6)
    >>> x = torch.randn(4, 32, 128)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 6])
    """

    def __init__(
        self,
        num_channels: int = 32,
        window_size: int = 128,
        num_classes: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        import torch.nn as nn

        self.input_proj = nn.Linear(window_size, d_model)
        self.pos_encoding = _PositionalEncoding(d_model, max_len=num_channels + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, channels, window_size)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        # x: (batch, channels, window_size) → project window_size to d_model
        x = self.input_proj(x)                          # (batch, channels, d_model)
        x = self.pos_encoding(x)                         # add positional encoding
        x = self.transformer_encoder(x)                  # (batch, channels, d_model)
        x = x.mean(dim=1)                                # global average pooling over channels
        return self.classifier(x)                        # (batch, num_classes)

    @classmethod
    def from_config(cls, cfg: object) -> "TransformerModel":
        """Instantiate a :class:`TransformerModel` from a :class:`~src.config.Config`.

        Parameters
        ----------
        cfg:
            A :class:`~src.config.Config` object.

        Returns
        -------
        TransformerModel
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
    ) -> "TransformerModel":
        """Load a :class:`TransformerModel` from a saved weights file.

        Parameters
        ----------
        weights_path:
            Path to the ``*.pt`` / ``*.pth`` file.
        num_channels, window_size, num_classes:
            Architecture parameters – must match the saved model.
        map_location:
            Passed to :func:`torch.load`.

        Returns
        -------
        TransformerModel
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
