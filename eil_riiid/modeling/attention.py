from typing import Optional, Tuple

import torch
import torch.nn as nn

AttentionContext = Tuple[torch.Tensor, torch.Tensor]
AttentionContext.__doc__ = """
    A group of attention context tensors.

    This class is a redefinition of a tuple of two tensors. It contains previously
    calculated key and value vectors from certain attention layer.

    They are used to the attention layers for attending the previous keys and values.
    Simply, they are cached to the contexts to reuse without re-calculating them.

    Note:
        Do not directly modify the attention contexts. It is sufficient to use the
        outputs from the attention layers.
"""


class BaseAttention(nn.Module):
    """An implementation of single attention mechanism.

    It first computes attention scores from query and key vectors. And then it performs
    a single-headed attention by attending values from the attention scores.

    Args:
        dropout_rate: A dropout rate. Note that a dropout layer is performed to the
            attention scores. Default is `0.1`.
        mask_value: A mask value for attention scores. The masked elements are replaced
            to this value and then the softmax is performed to the masked logits.
            Default is `-1e9`.
    """

    def __init__(self, dropout_rate: float = 0.1, mask_value: float = -1e9):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.mask_value = mask_value

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a single-headed attention.

        Args:
            q: The query tensor of shape (..., query_len, hidden_dims).
            k: The key tensor of shape (..., kv_len, hidden_dims).
            v: The value tensor of shape (..., kv_len, hidden_dims).
            mask: An optional boolean mask tensor of shape
                (..., query_len, hidden_dims). Default is `None`.

        Returns:
            The attended tensor of shape (..., query_len, hidden_dims).
        """
        x = torch.matmul(q, k.transpose(-1, -2)) / k.size(-1) ** 0.5

        if mask is not None:
            x = x + mask.type_as(x) * x.new_tensor(self.mask_value)
        x = self.dropout(x.softmax(-1))

        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    """An implementation of multi-headed attention mechanism.

    It splits query, key and value tensors into multi-heads and performs multi-headed
    attentions.

    Args:
        num_heads: The number of multi-heads.
        dropout_rate: A dropout rate. Note that a dropout layer is performed to the
            attention scores. Default is `0.1`.
        mask_value: A mask value for attention scores. The masked elements are replaced
            to this value and then the softmax is performed to the masked logits.
            Default is `-1e9`.
    """

    def __init__(
        self, num_heads: int, dropout_rate: float = 0.1, mask_value: float = -1e9
    ):
        super().__init__(dropout_rate, mask_value)
        self.num_heads = num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-headed attentions.

        Args:
            q: The query tensor of shape (..., query_len, hidden_dims).
            k: The key tensor of shape (..., kv_len, hidden_dims).
            v: The value tensor of shape (..., kv_len, hidden_dims).
            mask: An optional boolean mask tensor of shape
                (..., query_len, hidden_dims). Default is `None`.

        Returns:
            The attended tensor of shape (..., query_len, hidden_dims).
        """
        # Split tensors into multi-heads.
        q = q.view(q.size()[:-1] + (self.num_heads, -1))
        k = k.view(k.size()[:-1] + (self.num_heads, -1))
        v = v.view(v.size()[:-1] + (self.num_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Perform multi-headed attentions and merge them.
        return (
            super()
            .forward(q, k, v, mask)
            .transpose(-2, -3)
            .contiguous()
            .view(q.size()[:-3] + (q.size(-2), -1))
        )


class SelfAttentionLayer(nn.Module):
    """A self-attention layer with linear projections.

    This layer first performs linear projections to the input tensor. And it applies
    multi-headed attentions to the transformed query, key and value tensors. Finally, it
    performs a last linear projection to the attended tensor.

    Args:
        num_heads: The number of multi-heads.
        hidden_dims: The dimensionality of representation vectors.
        dropout_rate: A dropout rate. Note that a dropout layer is performed to the
            attention scores. Default is `0.1`.
        mask_value: A mask value for attention scores. The masked elements are replaced
            to this value and then the softmax is performed to the masked logits.
            Default is `-1e9`.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dims: int,
        dropout_rate: float = 0.1,
        mask_value: float = -1e9,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, dropout_rate, mask_value)
        self.proj_q = nn.Linear(hidden_dims, hidden_dims)
        self.proj_k = nn.Linear(hidden_dims, hidden_dims)
        self.proj_v = nn.Linear(hidden_dims, hidden_dims)
        self.proj_out = nn.Linear(hidden_dims, hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        ctx: Optional[AttentionContext] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, AttentionContext]:
        """Compute a self-attention with linear projections.

        Args:
            x: The input tensor of shape (..., seq_len, hidden_dims).
            ctx: An optional attention context tensors of shape
                (..., ctx_len, hidden_dims). Default is `None`.
            mask: An optional boolean mask tensor of shape
                (..., seq_len + ctx_len, hidden_dims). Default is `None`.

        Returns:
            - The attended tensor of shape (..., seq_len, hidden_dims).
            - A new attention context tensors of shape
                (..., seq_len + ctx_len, hidden_dims).
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # Reuse attention keys and values.
        if ctx is not None:
            k = torch.cat((ctx[0], k), dim=-2)
            v = torch.cat((ctx[1], v), dim=-2)

        x = self.proj_out(self.attn(q, k, v, mask))
        return x, (k, v)
