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
    High-Resolution STFT Audio MLP Backbone with configurable depth.
    Processes 2049-dimensional TKEO-STFT spectral vectors [B, 2049]:
      - Depth 1 (0 hidden layers): Linear(2049 -> 384) + LayerNorm(384) (~0.788M params)
      - Depth 2 (1 hidden layer):  Linear(2049 -> 512) + LN + GELU + Drop + Linear(512 -> 384) + LN (~1.248M params)
      - Depth 3 (2 hidden layers): Linear(2049 -> 768) + LN + GELU + Drop + Linear(768 -> 512) + LN + GELU + Drop + Linear(512 -> 384) + LN (~2.168M params)
      - Depth 4 (3 hidden layers): Linear(2049 -> 1024) + ... + Linear(512 -> 384) + LN (~3.482M params)
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
        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else ()
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Check if canonical 3-layer architecture for 100% exact state_dict compatibility
        self.is_canonical_3layer = (self.hidden_dims == (768, 512))

        if self.is_canonical_3layer:
            # Layer 1: in_features -> 768
            self.fc1 = nn.Linear(in_features, 768)
            self.ln1 = nn.LayerNorm(768)
            self.act1 = nn.GELU()
            self.drop1 = nn.Dropout(dropout)

            # Layer 2: 768 -> 512
            self.fc2 = nn.Linear(768, 512)
            self.ln2 = nn.LayerNorm(512)
            self.act2 = nn.GELU()
            self.drop2 = nn.Dropout(dropout)

            # Layer 3: 512 -> embed_dim
            self.fc3 = nn.Linear(512, embed_dim)
            self.ln3 = nn.LayerNorm(embed_dim)

            self._init_weights()
        else:
            # Dynamic depth support (Depth 1, Depth 2, Depth 4)
            self.blocks = nn.ModuleList()
            prev_dim = in_features
            for h_dim in self.hidden_dims:
                fc = nn.Linear(prev_dim, h_dim)
                init_layer(fc)
                ln = nn.LayerNorm(h_dim)
                act = nn.GELU()
                drop = nn.Dropout(dropout)
                self.blocks.append(nn.ModuleDict({"fc": fc, "ln": ln, "act": act, "drop": drop}))
                prev_dim = h_dim

            self.final_fc = nn.Linear(prev_dim, embed_dim)
            init_layer(self.final_fc)
            self.final_ln = nn.LayerNorm(embed_dim)

    def _init_weights(self) -> None:
        if self.is_canonical_3layer:
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

        if self.is_canonical_3layer:
            h1 = self.drop1(self.act1(self.ln1(self.fc1(x))))
            h2 = self.drop2(self.act2(self.ln2(self.fc2(h1))))
            f_audio = self.ln3(self.fc3(h2))
            return f_audio

        out = x
        for block in self.blocks:
            out = block["drop"](block["act"](block["ln"](block["fc"](out))))
        f_audio = self.final_ln(self.final_fc(out))
        return f_audio
