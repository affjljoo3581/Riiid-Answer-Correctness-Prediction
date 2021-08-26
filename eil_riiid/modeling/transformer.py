from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from eil_riiid.modeling.attention import AttentionContext, SelfAttentionLayer
from eil_riiid.modeling.embeddings import CategoricalEmbedding, PositionalEmbedding
from eil_riiid.modeling.feedforward import PositionwiseFeedForward
from eil_riiid.modeling.maskings import FutureMasking, PadMasking


class TransformerDecoderLayer(nn.Module):
    """An implementation of Transformer decoder layer.

    Transformer model is a stack of encoder and decoder layers. This class is an
    implementation of the Transformer decoder layer. It consists of a self-attention
    layer and a position-wise feed forward layer.

    Args:
        num_heads: The number of multi-heads for attention layer.
        hidden_dims: The dimensionality of representation vectors.
        bottleneck: An increment ratio of the dimensionality for the feed forward layer.
            Default is `0.1`.
        dropout_rate: The dropout rate for each layer. Default is `0.1`.
        mask_value: A mask value for attention scores. The masked elements are replaced
            to this value and then the softmax is performed to the masked logits.
            Default is `-1e9`.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dims: int,
        bottleneck: int = 4,
        dropout_rate: float = 0.1,
        mask_value: float = -1e9,
    ):
        super().__init__()
        self.attn = SelfAttentionLayer(num_heads, hidden_dims, dropout_rate, mask_value)
        self.ff = PositionwiseFeedForward(hidden_dims, bottleneck, dropout_rate)
        self.ln_attn = nn.LayerNorm(hidden_dims)
        self.ln_ff = nn.LayerNorm(hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        ctx: Optional[AttentionContext] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, AttentionContext]:
        """Perform the transformer decoder layer.

        Args:
            x: The input tensor of shape (..., seq_len, hidden_dims).
            ctx: An optional attention context tensors of shape
                (..., ctx_len, hidden_dims). Default is `None`.
            mask: An optional boolean mask tensor of shape
                (..., seq_len + ctx_len, hidden_dims). Default is
                `None`.

        Returns:
            - The output tensor of shape (..., seq_len, hidden_dims).
            - A new attention context tensors of shape
                (..., seq_len + ctx_len, hidden_dims).
        """
        a, ctx = self.attn(self.ln_attn(x), ctx, mask)
        x = x + a
        x = x + self.ff(self.ln_ff(x))
        return x, ctx


class Transformer(nn.Module):
    """An implementation of base Transformer model.

    Args:
        num_words: The number of embedding words.
        seq_len: The maximum input sequence length.
        pad_idx: The embedding index of padding token.
        num_layers: The number of transformer decoder layers.
        num_heads: The number of multi-heads for attention layers.
        hidden_dims: The dimensionality of representation vectors.
        bottleneck: An increment rate of the dimensionality for the feed forward layers.
            Default is `4`.
        dropout_rate: The dropout rate for each layer. Default is `0.1`.
        mask_value: A mask value for attention scores. The masked elements are replaced
            to this value and then the softmax is performed to the masked logits.
            Default is `-1e9`.
        bidirectional: A boolean whether to attend the tokens bidirectionaly or not.
            Default is `False`.
    """

    def __init__(
        self,
        num_words: int,
        seq_len: int,
        pad_idx: int,
        num_layers: int,
        num_heads: int,
        hidden_dims: int,
        bottleneck: int = 4,
        dropout_rate: float = 0.1,
        mask_value: float = -1e9,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        self.word_emb = CategoricalEmbedding(num_words, hidden_dims)
        self.position_emb = PositionalEmbedding(seq_len, hidden_dims)
        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    num_heads, hidden_dims, bottleneck, dropout_rate, mask_value
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_head = nn.LayerNorm(hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        ctxs: Optional[List[AttentionContext]] = None,
    ) -> Tuple[torch.Tensor, List[AttentionContext]]:
        """Perform the transformer model.

        Args:
            x: An input sequence tensors of shape (..., seq_len).
            aux: An optional auxiliary input tensor of shape
                (..., seq_len, hidden_dims).
            ctx_list: A list of attention context tensors of shape
                (..., ctx_len, hidden_dims).

        Returns:
            - The output tensor of shape (..., seq_len, hidden_dims)
            - A list of new attention context tensors of shape
                (..., seq_len + ctx_len, hidden_dims).
        """
        # Create masking tensors for attention layers.
        offset = ctxs[0][0].size(-2) if ctxs is not None else 0

        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Embed the inputs with auxiliary data and apply dropout.
        x = self.word_emb(x) + self.position_emb(x)
        if aux is not None:
            x = x + aux
        x = self.dropout(x)

        # Perform the transformer decoder layers sequentially.
        new_ctxs = []
        for i, decoder_layer in enumerate(self.decoder_layers):
            ctx = ctxs[i] if ctxs is not None else None
            x, ctx = decoder_layer(x, ctx, mask)
            new_ctxs.append(ctx)

        x = self.ln_head(x)
        return x, new_ctxs
