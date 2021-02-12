import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
import os


class LangDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        tokenizer: Tokenizer,
        x_label: str,
        y_label: str,
        max_data: int = 0
    ):
        super().__init__()

        assert os.path.exists(json_file), "json_file does not exists"

        self.tokenizer = tokenizer

        df = pd.read_json(json_file)
        if max_data > 0:
            df = df.sample(n=max_data)

        self.x = df[x_label].values if x_label and x_label in list(df.columns) else None
        self.has_y = False
        self.y = None
        self.labels = None

        if y_label and y_label in list(df.columns):
            self.has_y = True
            self.y = df[y_label].values

    def __len__(self):
        return len(self.x)

    def __getitem__(
        self,
        idx
    ):
        output = {}

        x = self.x[idx]
        output["input_texts"] = x
        output["input_ids"] = torch.tensor(self.tokenizer.encode(x).ids, dtype=torch.long)

        if self.has_y:
            y = self.y[idx]
            output["targets"] = torch.tensor(y, dtype=torch.long)

        return output
