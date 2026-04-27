"""Neural network models for EEG classification."""

from .cnn_model import CNNModel
from .transformer_model import TransformerModel

__all__ = ["CNNModel", "TransformerModel"]
