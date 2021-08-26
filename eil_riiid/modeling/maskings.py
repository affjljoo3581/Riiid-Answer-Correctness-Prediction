import torch
import torch.nn as nn


class PadMasking(nn.Module):
    """Mask padding tokens from attentions.

    Sequences are padded with padding tokens respectively to match the lengths equally.
    The padding tokens are meaningless and unnecessary as representations. They should
    be ignored from attentions. This class creates the corresponding mask tensor to mask
    the padding tokens in key and value sequences.

    Args:
        pad_idx: The embedding index of padding token.
    """

    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Create the mask tensor for padding tokens.

        Args:
            x: The input sequences tensor of shape (..., seq_len).
            offset: An offset of the input sequences. If the model uses attention
                contexts, the attention masks should be padded to covering the context
                tokens. Default is `0`.

        Returns:
            The mask tensor of shape (..., seq_len, seq_len + offset).
        """
        mask = (x == self.pad_idx).unsqueeze(-2)
        padding = x.new_zeros(x.size()[:-1] + (1, offset), dtype=torch.bool)

        mask = torch.cat((padding, mask), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])


class FutureMasking(nn.Module):
    """Mask posterior tokens from attentions.

    For language models, every predictions are dependent to their previous tokens. While
    training, all sequences include the future tokens for performance. In this case, the
    posterior tokens should be ignored from attentions. This class creates the
    corresponding mask tensor to mask the posterior tokens in key and value sequences.
    """

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Create the mask tensor for posterior tokens.

        Args:
            x: The input sequences tensor of shape (..., seq_len).
            offset: An offset of the input sequences. If the model uses attention
                contexts, the attention masks should be padded to covering the context
                tokens. Default is `0`.

        Returns:
            The mask tensor of shape (..., seq_len, seq_len + offset).
        """
        mask = x.new_ones((x.size(-1), x.size(-1) + offset), dtype=torch.bool)
        mask = mask.triu(offset + 1)

        mask = mask.view((1,) * (x.ndim - 1) + mask.size())
        mask = mask.expand(x.shape + mask.shape[-1:])

        return mask
