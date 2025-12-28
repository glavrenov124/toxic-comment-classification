from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TextCNNConfig:
    vocab_size: int
    embed_dim: int
    num_filters: int
    kernel_sizes: list[int]
    dropout: float
    num_labels: int


class TextCNN(nn.Module):
    def __init__(self, cfg: TextCNNConfig, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=pad_id)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=cfg.embed_dim,
                    out_channels=cfg.num_filters,
                    kernel_size=k,
                )
                for k in cfg.kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.num_filters * len(cfg.kernel_sizes), cfg.num_labels)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        emb = emb.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            x = torch.relu(conv(emb))
            x = torch.max(x, dim=2).values
            pooled.append(x)

        feats = torch.cat(pooled, dim=1)
        feats = self.dropout(feats)
        logits = self.fc(feats)
        return logits
