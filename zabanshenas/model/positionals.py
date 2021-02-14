import numpy as np
import torch
from torch import nn as nn


class Embedding(nn.Module):
    """ Page 5, Chapter 3.4, Embeddings and Softmax """

    def __init__(
        self,
        n_vocab: int,
        d_model: int,
        pad_idx: int = 0,
        scale_wte: bool = False
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.scale_wte = scale_wte

        self.wte = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
    ):
        # input_ids shape: (batch_size, seq_len)

        embeddings = self.wte(input_ids)

        if self.scale_wte:
            embeddings = embeddings * (self.d_model ** 0.5)

        # embeddings shape: (batch_size, seq_len, d_model)
        return embeddings


class FixedPositionalEncoding(nn.Module):
    """ 
    Page 5-6, Chapter 3.5, Fixed Positional Encoding 
    Page 7, Chapter 5.4, Regularization
    """

    def __init__(
        self,
        d_model: int,
        n_position: int = 5000,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            "positional_encoding",
            self._get_sinusoid_positional_encodeing(d_model, n_position)
        )

    def _get_sinusoid_positional_encodeing(
        self,
        d_model: int,
        n_position: int = 5000
    ):
        encodings = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position).unsqueeze(1)

        two_i = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(two_i * -(np.log(10000.0 / d_model)))

        encodings[:, 0::2] = torch.sin(position * div_term)  # 2i
        encodings[:, 1::2] = torch.cos(position * div_term)  # 2i+1

        # encodings = encodings.unsqueeze(1).requires_grad_(False)
        return encodings

    def forward(
        self,
        x: torch.Tensor
    ):
        # x shape: (batch_size, seq_len, d_model)
        # pe = self.positional_encoding[:x.shape[0]]
        pe = self.positional_encoding[:x.shape[1]]
        return self.dropout(x + pe)


class LearnedPositionalEncoding(nn.Module):
    """ 
    Page 5-6, Chapter 3.5, Fixed Positional Encoding 
    Page 7, Chapter 5.4, Regularization
    """

    def __init__(
        self,
        d_model: int,
        n_position: int = 5000,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # self.positional_encoding = nn.Parameter(torch.zeros(n_position, 1, d_model), requires_grad=True)
        self.positional_encoding = nn.Parameter(torch.zeros(n_position, d_model), requires_grad=True)

    def forward(
        self,
        x: torch.Tensor
    ):
        # x shape: (batch_size, seq_len, d_model)
        # pe = self.positional_encoding[:x.shape[0]]

        pe = self.positional_encoding[:x.shape[1]]
        return self.dropout(x + pe)


class PositionWiseFeedForward(nn.Module):
    """ Page 5, Chapter 3.3, Position-wise Feed-Forward Networks """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation=nn.ReLU(),
        bias1: bool = True,
        bias2: bool = True,
    ):
        super().__init__()

        self.L1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.activation = activation
        self.L2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor
    ):
        # x shape: (batch_size, seq_len, d_model)
        residual = x

        L1 = self.L1(x)
        L1 = self.activation(L1)

        L2 = self.L2(L1)
        L2 = self.dropout(L2)

        x = residual + L2
        x = self.norm(x)

        return x
