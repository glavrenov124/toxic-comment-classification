from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .utils import ensure_dir
from .vocab import Vocab, VocabConfig, build_vocab, tokenize


@dataclass(frozen=True)
class ProcessedPaths:
    train_path: str
    val_path: str
    test_path: str
    vocab_path: str


def download_raw_dataset(raw_dir: str, hf_name: str) -> None:
    raw_path = Path(raw_dir)
    if raw_path.exists() and any(raw_path.iterdir()):
        return

    ensure_dir(raw_dir)

    ds = load_dataset(
        "csv",
        data_files={"train": f"hf://datasets/{hf_name}/train.csv"},
    )

    ds["train"].save_to_disk(str(raw_path))


def load_from_disk_raw(raw_dir: str):
    return load_from_disk(raw_dir)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_splits_and_vocab(
    raw_dir: str,
    processed_dir: str,
    vocab_path: str,
    text_col: str,
    label_cols: list[str],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    lowercase: bool,
    vocab_size: int,
    min_freq: int,
) -> ProcessedPaths:
    processed_path = Path(processed_dir)
    train_path = processed_path / "train.jsonl"
    val_path = processed_path / "val.jsonl"
    test_path = processed_path / "test.jsonl"
    vocab_out = Path(vocab_path)

    if train_path.exists() and val_path.exists() and test_path.exists() and vocab_out.exists():
        return ProcessedPaths(
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            vocab_path=str(vocab_out),
        )

    ensure_dir(processed_dir)
    ensure_dir(str(vocab_out.parent))

    train_ds = load_from_disk_raw(raw_dir)
    df = train_ds.to_pandas()
    labels = df[label_cols].astype(int).values
    has_any = (labels.sum(axis=1) > 0).astype(int)

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    df_train, df_tmp, s_train, s_tmp = train_test_split(
        df,
        has_any,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=has_any,
    )

    rel_val = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_tmp,
        test_size=(1.0 - rel_val),
        random_state=seed,
        stratify=s_tmp,
    )

    vocab_cfg = VocabConfig(vocab_size=vocab_size, min_freq=min_freq, lowercase=lowercase)
    vocab = build_vocab(df_train[text_col].astype(str).tolist(), cfg=vocab_cfg)
    vocab.to_json(str(vocab_out))

    def to_rows(frame):
        rows = []
        for _, r in frame.iterrows():
            rows.append(
                {
                    "text": str(r[text_col]),
                    "labels": [int(r[c]) for c in label_cols],
                }
            )
        return rows

    _write_jsonl(train_path, to_rows(df_train))
    _write_jsonl(val_path, to_rows(df_val))
    _write_jsonl(test_path, to_rows(df_test))

    return ProcessedPaths(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        vocab_path=str(vocab_out),
    )


class ToxicDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Vocab, max_len: int, lowercase: bool):
        self.vocab = vocab
        self.max_len = max_len
        self.lowercase = lowercase

        self.samples: list[tuple[list[int], list[int]]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                tokens = tokenize(obj["text"], lowercase=self.lowercase)
                ids = vocab.encode(tokens)[: self.max_len]
                labels = obj["labels"]
                self.samples.append((ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        token_ids, labels = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


def collate_fn(batch, pad_id: int):
    token_ids, labels = zip(*batch, strict=False)
    lengths = [t.size(0) for t in token_ids]
    max_len = max(lengths)

    padded = []
    for t in token_ids:
        if t.size(0) < max_len:
            pad = torch.full((max_len - t.size(0),), pad_id, dtype=torch.long)
            padded.append(torch.cat([t, pad], dim=0))
        else:
            padded.append(t)

    return torch.stack(padded, dim=0), torch.stack(labels, dim=0)


def make_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    vocab: Vocab,
    batch_size: int,
    num_workers: int,
    max_len: int,
    lowercase: bool,
):
    train_ds = ToxicDataset(train_path, vocab=vocab, max_len=max_len, lowercase=lowercase)
    val_ds = ToxicDataset(val_path, vocab=vocab, max_len=max_len, lowercase=lowercase)
    test_ds = ToxicDataset(test_path, vocab=vocab, max_len=max_len, lowercase=lowercase)

    def _collate(b):
        return collate_fn(b, pad_id=vocab.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    return train_loader, val_loader, test_loader
