from __future__ import annotations

import torch
from torch import nn


class UrlLstmClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 64,
        hidden_size: int = 96,
        dropout_rate: float = 0.5,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * direction_multiplier, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, url_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(url_ids)
        _, (hidden, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            encoded = hidden[-1]
        return self.classifier(encoded).squeeze(-1)
