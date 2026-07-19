"""Building blocks that are byte-identical across all notebook families.

Keeping a single copy here is what prevents the silent divergence that
previously let the same class drift between notebooks.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, dim_feedforward=hidden_dim
        )
        self.fc = nn.Linear(embed_dim, num_classes+2)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, hidden_dim)
        )

    def forward(self, x, x_mask):
        # x: (batch_size, seq_len)
        B, L = x.shape
        x = self.embedding(x) + self.positional_encoding[:, :L, :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embed_dim)
        x = self.transformer(x, x)  # Using the same input as both src and tgt
        x = x.mean(dim=0)  # Pooling over the sequence dimension
        logits = self.fc(x)
        z = self.proj(x)              # (B, proj_dim)
        z = F.normalize(z, dim=1)
        return logits, z
