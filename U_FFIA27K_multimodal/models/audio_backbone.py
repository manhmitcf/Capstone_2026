import torch
import torch.nn as nn
from typing import Tuple, Optional


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


class AudioMLPBackbone(nn.Module):
    """
    High-Resolution STFT Audio MLP Backbone (AudioMLP_L2_compact_768_512, ~2.168M params).
    Processes 2049-dimensional TKEO-STFT spectral vectors [B, 2049]:
      - Layer 1: Linear(2049 -> 768) + LayerNorm(768) + GELU + Dropout(0.1)
      - Layer 2: Linear(768 -> 512) + LayerNorm(512) + GELU + Dropout(0.1)
      - Layer 3: Linear(512 -> 384) + LayerNorm(384) -> f_audio [B, 384]
    """
    def __init__(
        self,
        in_features: int = 2049,
        hidden_dims: Tuple[int, ...] = (768, 512),
        embed_dim: int = 384,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features = in_features

        # Handle backward-compatible scalar hidden_dim if provided
        if hidden_dim is not None and (hidden_dims == (768, 512) or not hidden_dims):
            hidden_dims = (hidden_dim,)
        self.hidden_dims = tuple(hidden_dims)
        self.embed_dim = embed_dim

        # Layer 1: in_features -> hidden_dims[0]
        h1 = self.hidden_dims[0] if len(self.hidden_dims) > 0 else 768
        self.fc1 = nn.Linear(in_features, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: hidden_dims[0] -> hidden_dims[1]
        h2 = self.hidden_dims[1] if len(self.hidden_dims) > 1 else 512
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Layer 3: hidden_dims[-1] -> embed_dim
        self.fc3 = nn.Linear(h2, embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT spectral feature vector [B, 2049]

        Returns:
            f_audio: Joint acoustic embedding [B, embed_dim]
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
            if x.size(-1) != self.in_features:
                x = x[:, :self.in_features]

        h1 = self.drop1(self.act1(self.ln1(self.fc1(x))))
        h2 = self.drop2(self.act2(self.ln2(self.fc2(h1))))
        f_audio = self.ln3(self.fc3(h2))
        return f_audio
