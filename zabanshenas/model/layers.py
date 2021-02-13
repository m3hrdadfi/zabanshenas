import torch
from torch import nn as nn
from zabanshenas.model.attentions import MultiHeadAttention
from zabanshenas.model.positionals import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """ Page 3, Chapter 3.1, Figure 1, Encoder Layer """

    def __init__(
        self,
        d_model: int,
        attention: MultiHeadAttention,
        feed_forward: PositionWiseFeedForward,
        keep_attention: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.keep_attention = keep_attention

        self.attention = attention
        self.feed_forward = feed_forward

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        # src, src_mask
        attention_list = None

        if self.keep_attention:
            x, attention_list = self.attention(
                query=src,
                key=src,
                value=src,
                mask=src_mask
            )

        else:
            x = self.attention(
                query=src,
                key=src,
                value=src,
                mask=src_mask
            )

        x = self.feed_forward(x)

        if self.keep_attention:
            return x, attention_list

        return x


class DecoderLayer(nn.Module):
    """ Page 3, Chapter 3.1, Figure 1, Decoder Layer """

    def __init__(
        self,
        d_model: int,
        attention: MultiHeadAttention,
        src_attention: MultiHeadAttention,
        feed_forward: PositionWiseFeedForward,
        keep_attention: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.keep_attention = keep_attention

        self.attention = attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        src: torch.Tensor,
        src_mask: torch.Tensor
    ):
        # tgt, tgt_mask
        # src, src_mask

        attention_list = None
        src_attention_list = None

        if self.keep_attention:
            x, attention_list = self.tgt_attention(
                query=tgt,
                key=tgt,
                value=tgt,
                mask=tgt_mask
            )
        else:
            x = self.tgt_attention(
                query=tgt,
                key=tgt,
                value=tgt,
                mask=tgt_mask
            )

        if self.keep_attention:
            x, src_attention_list = self.src_attention(
                query=x,
                key=src,
                value=src,
                mask=src_mask
            )
        else:
            x = self.src_attention(
                query=x,
                key=src,
                value=src,
                mask=src_mask
            )

        x = self.feed_forward(x)

        if self.keep_attention:
            return x, [attention_list, src_attention_list]

        return x
