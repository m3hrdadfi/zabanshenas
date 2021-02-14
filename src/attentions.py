from typing import Optional
import torch
from torch import nn as nn


class ScaledDotProductAttention(nn.Module):
    """ Page 4, Chapter 3.2.1, Scaled Dot-Product Attention """

    def __init__(
        self,
        scaler: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.scaler = scaler
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        # attention shape: (batch_size, heads, seq_len, seq_len)
        # query shape: (batch_size, heads, seq_len, d_k)
        # key shape: (batch_size, heads, seq_len, d_k) -> (batch_size, heads, d_k, seq_len)
        # -> (batch_size, heads, seq_len, seq_len)
        attention = torch.matmul(query * self.scaler, key.transpose(-2, -1))

        if mask is not None:
            attention = attention.masked_fill(mask == torch.tensor(False), -1e7)

        attention = self.dropout(self.softmax(attention))

        # attention shape: (batch_size, heads, seq_len, seq_len)
        # value shape: (batch_size, heads, seq_len, d_v)
        # output shape: (batch_size, heads, seq_len, d_v)
        output = torch.matmul(attention, value)

        return output, attention


class MultiHeadAttention(nn.Module):
    """ Page 4-5, Chapter 3.2.2, Multi-Head Attention """

    def __init__(
        self,
        d_model: int,
        heads: int,
        keep_attention: bool = False,
        dropout_rate: int = 0.1
    ):
        super().__init__()

        assert d_model % heads == 0, "`d_model` needs to be divisible by `heads`"

        self.d_model = d_model
        self.heads = heads
        self.keep_attention = keep_attention

        d_k = d_v = d_model // heads
        self.d_k = d_k
        self.d_v = d_v
        self.scale = 1 / (d_k ** 0.5)

        self.W_Q = nn.Linear(d_model, heads * d_k)
        self.W_K = nn.Linear(d_model, heads * d_k)
        self.W_V = nn.Linear(d_model, heads * d_v)
        self.fc = nn.Linear(heads * d_v, d_model)

        self.attention = ScaledDotProductAttention(d_k ** 0.5)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        # query shape: (batch_size, seq_len, heads x d_k)
        # key shape: (batch_size, seq_len, heads x d_k)
        # value shape: (batch_size, seq_len, heads x d_v)

        batch_size = query.shape[0]
        residual = query

        # Attention projection
        # query shape: (batch_size, seq_len, heads x d_k) => (batch_size, seq_len, heads, d_k) => (batch_size, heads, seq_len, d_k)
        # key shape: (batch_size, seq_len, heads x d_k) => (batch_size, seq_len, heads, d_k) => (batch_size, heads, seq_len, d_k)
        # value shape: (batch_size, seq_len, heads x d_v) => (batch_size, seq_len, heads, d_k) => (batch_size, heads, seq_len, d_v)
        query = self.W_Q(query).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        key = self.W_K(key).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        value = self.W_V(value).view(batch_size, -1, self.heads, self.d_v).transpose(1, 2)

        x, attention = self.attention(query, key, value, mask=mask)

        # x (context) shape: (batch_size, heads, seq_len, d_v)
        # x reshape steps:
        # (batch_size, heads, seq_len, d_v) -> (batch_size, seq_len, heads, d_v)
        # (batch_size, seq_len, heads, d_v) -> (batch_size, seq_len, heads x d_v)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * self.d_v)
        x = self.dropout(self.fc(x))

        x = residual + x
        x = self.norm(x)

        if self.keep_attention:
            return x, attention.detach()

        return x
