from __future__ import annotations

import torch
from torch import nn


class HtmlCnnClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 64,
        conv_filters: int = 128,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, html_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(html_ids).transpose(1, 2)
        encoded = self.encoder(embedded)
        return self.classifier(encoded).squeeze(-1)
