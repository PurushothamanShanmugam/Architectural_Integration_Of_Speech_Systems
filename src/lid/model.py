import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLID(nn.Module):
    def __init__(self, n_mels: int = 40, hidden: int = 256, n_layers: int = 3, n_heads: int = 8, n_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(n_mels, 128)
        self.lstm = nn.LSTM(128, hidden, n_layers, batch_first=True, bidirectional=True, dropout=0.3)
        d_model = hidden * 2
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(x))
        x, _ = self.lstm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.head(x)
