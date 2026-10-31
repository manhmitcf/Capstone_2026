import torch
import torch.nn as nn
from typing import Tuple


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Non-blocking Bernoulli implementation avoiding GPU-to-CPU synchronization stalls.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Zero trainable parameters; identity pass-through during evaluation mode.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (Liu et al., CVPR 2022).
    - Depthwise Conv 7x7
    - LayerNorm (across channels)
    - Pointwise Conv / Linear (dim -> 4*dim)
    - GELU activation
    - Pointwise Conv / Linear (4*dim -> dim)
    - DropPath (Stochastic Depth)
    - Residual Connection
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        # Permute for LayerNorm & Linear: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return shortcut + self.drop_path(x)


class ConvNeXtNanoVideoBackbone(nn.Module):
    """
    Streamlined ConvNeXt-Nano Video Backbone tailored for 6-channel kinematic inputs across T=2 frames (C6_rgb_flow_vorticity).
    Channels:
      - 0, 1, 2: Spatial RGB appearance
      - 3, 4:    Optical Flow (u, v) swimming velocity
      - 5:       Fluid Vorticity omega = dv/dx - du/dy

    Architecture:
      - Stem: Conv 4x4 (6 -> 48) + GroupNorm (4,752 params)
      - Stage 1: 48ch  x 1 block  (DropPath linear schedule)
      - Stage 2: 96ch  x 1 block  (DropPath linear schedule)
      - Stage 3: 192ch x 3 blocks (DropPath linear schedule)
      - Stage 4: 384ch x 1 block  (DropPath linear schedule)
      - Temporal Aggregation (composite_ln default): LN((v0 + v1) + |v1 - v0|)
      - Native 384-dim: Direct 384-dimensional representation preserving full channel capacity.
      - Total parameters: 2,614,176 params (composite_ln) / 2,613,408 params (sum, abs_diff).
    """
    def __init__(
        self,
        embed_dim: int = 384,
        in_chans: int = 6,
        dims: Tuple[int, ...] = (48, 96, 192, 384),
        depths: Tuple[int, ...] = (1, 1, 3, 1),
        num_frames: int = 2,
        drop_path_rate: float = 0.1,
        temporal_mode: str = "sum",
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.drop_path_rate = drop_path_rate
        self.temporal_mode = temporal_mode
        assert embed_dim == dims[-1], (
            f"embed_dim ({embed_dim}) must match backbone stage-4 dimension ({dims[-1]})"
        )

        # 1. Stem: Patchify 4x4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6)  # Equivalent to LayerNorm over [C, H, W]
        )

        # 2. Downsample layers between stages
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        # 3. Stages with Stochastic Depth (Linear Schedule)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                )
                for j in range(depths[i])
            ])
            cur += depths[i]
            self.stages.append(stage)

        # 4. Final normalization (native 384-dim)
        self.norm_final = nn.LayerNorm(dims[-1], eps=1e-6)

        # 5. Temporal composite LayerNorm
        self.norm_composite = nn.LayerNorm(embed_dim, eps=1e-6) if temporal_mode == 'composite_ln' else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature vector from [B * T, C, H, W] -> [B * T, 384].
        """
        x = self.stem(x)
        for i in range(4):
            if i > 0:
                x = self.downsample_layers[i - 1](x)
            x = self.stages[i](x)

        # Global Average Pooling: [B * T, 384, H', W'] -> [B * T, 384]
        x = x.mean(dim=[-2, -1])
        x = self.norm_final(x)
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W] tensor (C=6, T=2)

        Returns:
            f_video: Joint spatiotemporal video embedding [B, embed_dim]
        """
        B, T, C, H, W = frames.shape

        # Process all T frames: [B * T, C, H, W] -> [B * T, embed_dim]
        flat_frames = frames.reshape(B * T, C, H, W)
        flat_tokens = self.forward_features(flat_frames)  # [B * T, embed_dim]

        # Reshape to temporal sequence of frame tokens: [B, T, embed_dim]
        tokens_video = flat_tokens.view(B, T, self.embed_dim)

        if T < 2:
            return self.norm_composite(tokens_video[:, 0]) if self.norm_composite is not None else tokens_video[:, 0]

        if self.temporal_mode == "composite_ln":
            v_sum = tokens_video[:, 0] + tokens_video[:, 1]
            v_diff = torch.abs(tokens_video[:, 1] - tokens_video[:, 0])
            f_video = self.norm_composite(v_sum + v_diff)
        elif self.temporal_mode == "sum":
            f_video = tokens_video[:, 0] + tokens_video[:, 1]
        elif self.temporal_mode == "abs_diff":
            f_video = torch.abs(tokens_video[:, 1] - tokens_video[:, 0])
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}. Available: 'composite_ln', 'sum', 'abs_diff'.")

        return f_video
