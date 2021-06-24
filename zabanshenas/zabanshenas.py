from typing import Any, Dict, Optional
import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from zabanshenas.normalizer import Normalizer
from zabanshenas.languages import languages


class Zabanshenas:
    def __init__(
            self,
            model_name_or_path: str = "m3hrdadfi/zabanshenas-roberta-base-mix",
            by_gpu: bool = False
    ) -> None:
        self.device = torch.device("cpu" if not by_gpu else "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        self.normalizer = Normalizer()
        self.languages = languages
        self.framework = "pt"
        self.max_length = 512

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        """

        return {
            name: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }

    def _parse_and_tokenize(
            self,
            inputs,
            do_normalization: bool = True,
            max_length: int = 512,
            padding: bool = True,
            add_special_tokens: bool = True,
            truncation: bool = True,
    ):
        """
        Parse arguments and tokenize
        """
        inputs = [self.normalizer(item) for item in inputs]
        max_length = min(max_length, self.max_length)
        inputs = self.tokenizer(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
            truncation=truncation,
        )

        return inputs

    def _forward(
            self,
            inputs,
            return_tensors: bool = True
    ):
        with torch.no_grad():
            inputs = self.ensure_tensor_on_device(**inputs)
            predictions = self.model(**inputs)[0].cpu()

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()

    def detect(
            self,
            texts,
            max_length: int = 128,
            do_normalization: bool = True,
            return_all_scores: bool = False,
            return_all_scores_sorted: bool = True
    ):
        texts = [texts] if not isinstance(texts, list) else texts
        inputs = self._parse_and_tokenize(texts, do_normalization=do_normalization, max_length=max_length)
        outputs = self._forward(inputs, return_tensors=False)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)

        if return_all_scores:
            results = [
                [
                    {
                        "language": self.languages.get(self.model.config.id2label[i], None),
                        "code": self.model.config.id2label[i],
                        "score": score.item()
                    } for i, score in enumerate(item)
                ] for item in scores
            ]
            if return_all_scores_sorted:
                results = [list(sorted(result, key=lambda kv: kv["score"], reverse=True)) for result in results]
        else:
            results = [
                {
                    "language": self.languages.get(self.model.config.id2label[item.argmax()], None),
                    "code": self.model.config.id2label[item.argmax()],
                    "score": item.max().item()
                } for item in scores
            ]

        return results
