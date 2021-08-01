import streamlit as st

from typing import Any, Dict, Optional
import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from libs.normalizer import Normalizer
from libs.languages import languages
from libs.examples import EXAMPLES
from libs.dummy import outputs as dummy_outputs
from libs.utils import plot_result

import meta


class Zabanshenas:
    def __init__(
            self,
            model_name_or_path: str = "m3hrdadfi/zabanshenas-roberta-base-mix",
            by_gpu: bool = False
    ) -> None:
        self.debug = True
        self.dummy_outputs = dummy_outputs
        self.device = torch.device("cpu" if not by_gpu else "cuda")
        self.model_name_or_path = model_name_or_path

        self.tokenizer = None
        self.model = None
        self.normalizer = None
        self.languages = None
        self.framework = "pt"
        self.max_length = 512
        self.top_k = 5

    def load(self):
        if not self.debug:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(self.device)

        self.normalizer = Normalizer()
        self.languages = languages

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
            do_normalization: bool = True
    ):
        if self.debug:
            return self.dummy_outputs

        texts = [texts] if not isinstance(texts, list) else texts
        inputs = self._parse_and_tokenize(texts, do_normalization=do_normalization, max_length=max_length)
        outputs = self._forward(inputs, return_tensors=False)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)

        results = [
            [
                {
                    "language": self.languages.get(self.model.config.id2label[i], None),
                    "code": self.model.config.id2label[i],
                    "score": score.item()
                } for i, score in enumerate(item)
            ] for item in scores
        ]
        results = [list(sorted(result, key=lambda kv: kv["score"], reverse=True)) for result in results]

        return results


@st.cache(allow_output_mutation=True)
def load_language_detector():
    detector = Zabanshenas()
    detector.load()
    return detector


def main():
    st.set_page_config(
        page_title="Zabanshenas",
        page_icon="🕵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    detector = load_language_detector()

    col1, col2 = st.beta_columns([6, 4])
    with col2:
        st.markdown(meta.INFO, unsafe_allow_html=True)

    with col1:
        prompts = list(EXAMPLES.keys()) + ["Custom"]
        prompt = st.selectbox(
            'Examples (select from this list)',
            prompts,
            # index=len(prompts) - 1,
            index=0
        )

        if prompt == "Custom":
            prompt_box = ""
        else:
            prompt_box = EXAMPLES[prompt]

        text = st.text_area(
            'Insert your text: ',
            detector.normalizer(prompt_box),
            height=200
        )
        text = detector.normalizer(text)
        entered_text = st.empty()

    detect_language = st.button('Detect Language !')

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    if detect_language:
        words = text.split()
        with st.spinner("Detecting..."):
            if not len(words) > 3:
                entered_text.markdown(
                    "Insert your text (at least three words)"
                )
            else:
                top_languages = detector.detect(text, max_length=min(len(words), detector.max_length))
                top_languages = top_languages[0][:detector.top_k]
                plot_result(top_languages)


if __name__ == '__main__':
    main()
