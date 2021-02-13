from typing import Optional

import torch
from torch.nn import functional as F
from torch import nn as nn

from zabanshenas.model.attentions import MultiHeadAttention
from zabanshenas.model.layers import EncoderLayer
from zabanshenas.model.positionals import (
    Embedding,
    PositionWiseFeedForward,
    FixedPositionalEncoding as PositionalEncoding
)

import copy
import os

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

from zabanshenas.utils import make_src_mask


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        n_layers: int,
        ff_d_ff: int,
        ff_activation: nn.Module = nn.ReLU(),
        ff_bias1: bool = True,
        ff_bias2: bool = True,
        dropout_rate: float = 0.1,
        keep_attention: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.ff_d_ff = ff_d_ff
        self.n_layers = n_layers
        self.ff_activation = ff_activation
        self.ff_bias1 = ff_bias1
        self.ff_bias2 = ff_bias2
        self.keep_attention = keep_attention

        attention_layer = MultiHeadAttention(
            d_model=d_model,
            heads=heads,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )
        ff_layer = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=ff_d_ff,
            dropout_rate=dropout_rate,
            activation=ff_activation,
            bias1=ff_bias1,
            bias2=ff_bias2,
        )
        encoder_layer = EncoderLayer(
            d_model=d_model,
            attention=attention_layer,
            feed_forward=ff_layer,
            keep_attention=keep_attention
        )
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor
    ):
        # src shape: (batch_size, seq_len, d_model)
        # src_mask shape: (batch_size, 1, 1, seq_len)
        attention_list_layers = []

        x = src
        for layer in self.layers:
            if self.keep_attention:
                x, attention_list = layer(src=x, src_mask=src_mask)
                attention_list_layers.extend([attention_list])
            else:
                x = layer(src=x, src_mask=src_mask)

        x = self.norm(x)

        if self.keep_attention:
            return x, attention_list_layers

        return x


class TransformerLangDetection(nn.Module):

    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
        d_model: int,
        heads: int,
        ff_d_ff: int,
        n_layers: int,
        n_classes: int,
        n_position: int = 5000,
        ff_activation: nn.Module = nn.ReLU(),
        ff_bias1: bool = True,
        ff_bias2: bool = True,
        dropout_rate: float = 0.1,
        scale_wte: bool = False,
        keep_attention: bool = False
    ):
        super().__init__()

        self.keep_attention = keep_attention

        self.wte = Embedding(n_vocab, d_model, pad_idx=pad_idx, scale_wte=scale_wte)
        self.pe = PositionalEncoding(d_model, n_position, dropout_rate=dropout_rate)
        self.encoder = Encoder(
            d_model=d_model,
            heads=heads,
            n_layers=n_layers,
            ff_d_ff=ff_d_ff,
            ff_activation=ff_activation,
            ff_bias1=ff_bias1,
            ff_bias2=ff_bias2,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )
        self.classifer = nn.Linear(d_model, n_classes)

        self.model_args = {
            "n_vocab": n_vocab,
            "pad_idx": pad_idx,
            "d_model": d_model,
            "heads": heads,
            "ff_d_ff": ff_d_ff,
            "n_layers": n_layers,
            "n_classes": n_classes,
            "n_position": n_position,
            "ff_activation": ff_activation,
            "ff_bias1": ff_bias1,
            "ff_bias2": ff_bias2,
            "dropout_rate": dropout_rate,
            "scale_wte": scale_wte,
            "keep_attention": keep_attention,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
    ):
        attention_list_layers = []

        embedding = self.wte(input_ids)
        pe = self.pe(embedding)

        if self.keep_attention:
            encoded, attention_list_layers = self.encoder(pe, input_mask)
        else:
            encoded = self.encoder(pe, input_mask)

        mean_encoded = torch.mean(encoded, dim=1)
        logits = self.classifer(mean_encoded)

        if self.keep_attention:
            return logits, attention_list_layers

        return logits

    @staticmethod
    def load_tokenizer(
        tokenizer_file: str,
        enable_padding: bool = True,
        pad_token: str = "<pad>",
        max_seq_len: int = 512
    ):
        assert os.path.exists(tokenizer_file), "tokenizer_file does not exists"
        tokenizer = Tokenizer.from_file(tokenizer_file)

        if enable_padding:
            tokenizer.enable_padding(
                direction="right",
                pad_id=tokenizer.encode(pad_token).ids[0],
                pad_type_id=0,
                pad_token=pad_token,
                length=max_seq_len,
                pad_to_multiple_of=None
            )

            tokenizer.enable_truncation(
                max_seq_len,
                stride=0,
                strategy='longest_first'
            )

        return tokenizer

    def save(
        self,
        model_dir: str,
    ):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, "model.bin"))
        torch.save(self.model_args, os.path.join(model_dir, "model_args.bin"))

    @staticmethod
    def load(
        model_dir: str,
        model_args: dict = {},
        device="cpu"
    ):
        assert os.path.exists(model_dir), "model_dir does not exists"
        device = torch.device(device)

        try:
            model_args_default = torch.load(
                os.path.join(model_dir, "model_args.bin"),
                map_location=device
            )
            model_args_default.update(model_args)

            model = TransformerLangDetection(**model_args_default).to(device)
            model.load_state_dict(torch.load(
                os.path.join(model_dir, "model.bin"),
                map_location=device
            ))
        except Exception as e:
            print(e)
            model = None

        return model

    def predict(
        self,
        x: list,
        tokenizer: Tokenizer,
        label_map: dict = {},
        batch_size: int = 32,
        progressbar: bool = True,
        keep_attention: bool = False,
        pad_token: str = "<pad>",
        topk: int = 5,
        device: str = "cpu"
    ):
        device = torch.device(device)

        n_x = len(x)
        n_batches = int(np.ceil(n_x / batch_size)) if n_x > batch_size else 1
        has_label_map = True if isinstance(label_map, list) and len(label_map) > 0 else False

        attention_list = None
        input_tokens_list = []

        y_preds = []
        y_names = []
        y_probs = []

        topk_preds = []
        topk_probs = []

        model = self
        model = model.to(device)

        keep_attention = True if model.keep_attention or keep_attention else False

        for i in tqdm(range(n_batches), disable=not progressbar, position=0):
            batch = x[i * batch_size: (i + 1) * batch_size]

            input_ids = torch.tensor([tokenizer.encode(text).ids for text in batch], dtype=torch.long)
            input_masks = make_src_mask(input_ids)

            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)

            if keep_attention:
                input_tokens = [[tokenizer.decode([token_id]) for token_id in tokenizer.encode(
                    text).ids if token_id != tokenizer.encode(pad_token).ids[0]] for text in batch]
                input_tokens_list.extend(input_tokens)

            model.eval()
            with torch.no_grad():

                if keep_attention:
                    outputs, _attention_list = model(input_ids, input_masks)
                    attention_list = [attention.detach().cpu().numpy() for attention in _attention_list]
                else:
                    outputs = model(input_ids, input_masks)

                probs = F.softmax(outputs, dim=1)
                probs, preds = torch.topk(probs, k=topk)

                probs = probs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()

                y_probs.extend(probs[:, 0].tolist())
                y_preds.extend(preds[:, 0].tolist())

                topk_probs.extend(probs.tolist())
                topk_preds.extend(preds.tolist())

                if has_label_map:
                    y_names.extend([label_map[pred] for pred in preds[:, 0].tolist()])

        y_probs = np.array(y_probs)
        y_preds = np.array(y_preds)

        topk_probs = np.array(topk_probs)
        topk_preds = np.array(topk_preds)

        if keep_attention:

            if has_label_map:
                return [y_probs, y_preds, y_names], [topk_probs, topk_preds], attention_list, input_tokens_list
            else:
                return [y_probs, y_preds], [topk_probs, topk_preds], attention_list, input_tokens_list

        if has_label_map:
            return [y_probs, y_preds, y_names], [topk_probs, topk_preds]

        return [y_probs, y_preds], [topk_probs, topk_preds]
