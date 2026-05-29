from __future__ import annotations

import torch
from torch import nn


class DualBranchCnnClassifier(nn.Module):
    def __init__(
        self,
        url_vocab_size: int = 128,
        html_vocab_size: int = 256,
        embedding_dim: int = 64,
        conv_filters: int = 128,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.url_embedding = nn.Embedding(url_vocab_size, embedding_dim, padding_idx=0)
        self.html_embedding = nn.Embedding(html_vocab_size, embedding_dim, padding_idx=0)
        self.url_encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.html_encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(conv_filters * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, url_ids: torch.Tensor, html_ids: torch.Tensor) -> torch.Tensor:
        url_embedded = self.url_embedding(url_ids).transpose(1, 2)
        html_embedded = self.html_embedding(html_ids).transpose(1, 2)
        url_vector = self.url_encoder(url_embedded)
        html_vector = self.html_encoder(html_embedded)
        fused = torch.cat((url_vector, html_vector), dim=1)
        return self.fusion(fused).squeeze(-1)
