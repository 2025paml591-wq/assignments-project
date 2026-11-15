import re
from functools import lru_cache

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.,:/'-][A-Za-z0-9]+)*|[\u0900-\u097f]+")


class Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding = None

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> list:
        try:
            if self._encoding is None:
                import tiktoken

                self._encoding = tiktoken.get_encoding(self.encoding_name)
            return self._encoding.encode(text)
        except ImportError:
            return text.split()

    def decode(self, tokens: list) -> str:
        try:
            if self._encoding is None:
                import tiktoken

                self._encoding = tiktoken.get_encoding(self.encoding_name)
            return self._encoding.decode(tokens)
        except ImportError:
            return " ".join(tokens)

    def split(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    return Tokenizer()
