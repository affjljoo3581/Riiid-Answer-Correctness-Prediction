import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish: A Self-Gated Activation Function"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Sequential):
    """A position-wise feed-forward layer.

    This layer consists of two dense layers with `Swish` activation and dropout layer.
    After performing a self-attention layer, the representations are transformed through
    this position-wise feed-forward layer.

    Args:
        hidden_dims: The dimensionality of the representation vectors.
        bottleneck: An increment rate of the dimensionality in the first dense layer.
            Default is `4`.
        dropout_rate: The dropout rate. Note that the dropout layer is performed after
            the first dense layer. Default is `0.1`.
    """

    def __init__(
        self, hidden_dims: int, bottleneck: int = 4, dropout_rate: float = 0.1
    ):
        super().__init__(
            nn.Linear(hidden_dims, hidden_dims * bottleneck),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims * bottleneck, hidden_dims),
        )
