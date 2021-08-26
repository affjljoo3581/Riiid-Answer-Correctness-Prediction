import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Embedding):
    """A discrete-position embedding layer.

    Transformer-based models need an order-information of input sequences due to the
    non-recursiveness. This layer creates the discrete position information by embedding
    the position indices to their representation vectors.

    Note:
        The `num_embeddings` parameter must be set to the maximum sequence length, not
        the vocabulary size.
    """

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Map the positions to the embedding vectors.

        Args:
            x: The input tensor of shape (..., seq_len).
            offset: An offset of the input sequences. If the model uses attention
                contexts, the position information must be shifted for the correct
                absolute order. Default is `0`.

        Returns:
            The embedding tensor of shape (..., seq_len, embedding_dim).
        """
        p = torch.arange(offset, offset + x.size(-1), dtype=torch.long, device=x.device)
        p = super().forward(p)

        p = p.view((1,) * (x.ndim - 1) + p.size())
        return p.expand(x.size() + (-1,))

    def reset_parameters(self):
        """Modify the initialization of embedding parameters."""
        nn.init.normal_(self.weight, std=0.02)


class ContinuousEmbedding(nn.Module):
    """A continuous-data embedding layer.

    Basically, language models are designed to handle categorical discrete data. For
    feeding continuous values to the model, they also be encoded to the representation
    vectors. This layer uses sinusoidal positional encoding to encode the continuous
    data. It calculates the sinusoidal encodings for the data and multiplies to a
    learnable parameter.

    Args:
        hidden_dims: The dimensionality of the encoded vectors.
    """

    def __init__(self, hidden_dims: int):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.weight = nn.Parameter(torch.empty(hidden_dims).normal_(std=0.02))

        x = torch.arange(0, hidden_dims, 2, dtype=torch.float)
        x = (-x * math.log(10000) / hidden_dims).exp()
        self.register_buffer("basis", x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the continuous data to the computable vectors.

        Args:
            x: The input tensor of shape (..., seq_len).

        Returns:
            The encoded tensor of shape (..., seq_len, hidden_dims).
        """
        x = self.basis.unsqueeze(-2) * x.unsqueeze(-1)
        x = torch.stack((torch.sin(x), torch.cos(x)), dim=-1)

        x = self.weight * x.view(x.size()[:-2] + (-1,))
        return x.type_as(self.weight)


class CategoricalEmbedding(nn.Embedding):
    """A discrete categorical data embedding layer.

    It embeds categorical discrete tokens (e.g. characters, words and subwords) to their
    representation vectors. It is used to feed the discrete sequential data to a model.
    """

    def forward(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Embeds the categorical tokens to representation vectors.

        Args:
            x: The input tensor of shape (..., seq_len) if `transposed == True` else
                (..., seq_len, embedding_dim).
            transpose: A boolean determining whether to embeds the tokens or project to
                the category space.

        Returns
            The output tensor of shape (..., seq_len, embedding_dim) if
            `transposed == True` else (..., seq_len, num_embeddings).
        """
        if transpose:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)

    def reset_parameters(self):
        """Modify the initialization of embedding parameters."""
        nn.init.normal_(self.weight, std=0.02)
