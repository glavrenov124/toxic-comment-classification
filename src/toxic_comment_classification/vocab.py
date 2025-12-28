from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class VocabConfig:
    vocab_size: int
    min_freq: int
    lowercase: bool


class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> Vocab:
        with open(path, encoding="utf-8") as f:
            token_to_id = json.load(f)
        return cls(token_to_id=token_to_id)


def tokenize(text: str, lowercase: bool) -> list[str]:
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def build_vocab(texts: Iterable[str], cfg: VocabConfig) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text, cfg.lowercase))

    most_common = [
        token for token, freq in counter.most_common(cfg.vocab_size) if freq >= cfg.min_freq
    ]

    token_to_id = {Vocab.PAD: 0, Vocab.UNK: 1}
    for token in most_common:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    return Vocab(token_to_id=token_to_id)
