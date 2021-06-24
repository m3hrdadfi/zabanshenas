import re
import regex
import sys
import textwrap
from typing import Any, Dict, Optional

punctuations = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
    '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
    '`', '{', '|', '}', '~', '»', '«', '“', '”', "-",
]


class Normalizer:
    """A general normalizer for every language"""

    _whitelist = r"[" + "\p{N}\p{L}\p{M}" + re.escape("".join(punctuations)) + "]+"
    _dictionary = {}

    def __init__(
            self,
            whitelist: str = None,
            dictionary: Dict[str, str] = None,
    ) -> None:
        self.whitelist = whitelist if whitelist and isinstance(whitelist, str) else self._whitelist
        self.dictionary = dictionary if dictionary and isinstance(dictionary, dict) else self._dictionary

    def chars_to_map(self, sentence: str) -> str:
        """Maps every character, words, and phrase into a proper one.

        Args:
            sentence (str): A piece of text.
        """
        if not len(self.dictionary) > 0:
            return sentence

        pattern = "|".join(map(re.escape, self.dictionary.keys()))
        return re.sub(pattern, lambda m: self.dictionary[m.group()], str(sentence))

    def chars_to_preserve(
            self,
            sentence: str,
    ) -> str:
        """Keeps specified characters from sentence

        Args:
            sentence (str): A piece of text.
        """
        try:
            tokenized = regex.findall(self.whitelist, sentence)
            return " ".join(tokenized)
        except Exception as error:
            print(
                textwrap.dedent(
                    f"""
                    Bad characters range {self.whitelist},
                    {error}
                    """
                )
            )
            raise

    def text_level_normalizer(self, text: str) -> str:
        """A text level of normalization"""

        text = regex.sub(r"([" + re.escape("".join(punctuations)) + "])", r" \1 ", text)
        text = text.strip()

        return text

    def __call__(
            self,
            text: str,
            do_lowercase: Optional[bool] = False
    ) -> Any:
        """Normalization caller"""

        text = self.chars_to_map(text)
        text = self.chars_to_preserve(text)
        text = self.text_level_normalizer(text)
        text = re.sub(r"\s+", " ", text)

        if do_lowercase:
            text = text.lower()

        return text
