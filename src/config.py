import dataclasses
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

import torch
from torch import nn as nn

import json



@dataclass
class BaseArguments:
    def to_dict(self):
        """ Serializes this instance """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """ Sanitized serialization """
        d = self.to_dict()

        valid_types = [bool, int, float, str]
        valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    def to_json(self, indent=2):
        """ Jsonifies this instance """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class DataTrainingArguments(BaseArguments):
    train_file: str = "../data/train.json"
    valid_file: str = "../data/validation.json"
    tokenizer_file: str = "../config/tokenizer.json"
    x_label: str = "text"
    y_label: str = "lang_id"
    pad_token: str = "<pad>"
    enable_padding: bool = True
    max_seq_len: int = 512
    train_batch_size: int = 8
    train_shuffle: bool = True
    valid_batch_size: int = 8
    valid_shuffle: bool = False


@dataclass
class ModelArguments(BaseArguments):
    d_model: int = 512
    heads: int = 8
    ff_d_ff: int = 1024
    n_position: int = 5000
    n_layers: int = 3
    n_classes: int = 235
    ff_activation = nn.ReLU()
    ff_bias1: bool = True
    ff_bias2: bool = True
    dropout_rate: float = 0.1
    scale_wte: bool = True
    keep_attention: bool = False


@dataclass
class TrainingArguments(BaseArguments):
    learning_rate: float = 1e-4
    log_interval: int = 2000
    n_epochs: int = 5
    checkpoint_dir: str = "./ckpts/"
    n_checkpoints: int = 3
