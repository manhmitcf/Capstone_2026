import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FishMotionKinematics6Ch(nn.Module):
    """
    Differentiable 6-Channel Motion Kinematics Extractor for Fish Feeding Intensity Assessment (C6_rgb_flow_vorticity):
      - Channels 0-2 : Spatial RGB appearance (fish clustering, surface foam, feeding pellets)
      - Channels 3-4 : Dense Optical Flow (u, v) representing horizontal and vertical swimming velocities
      - Channel 5    : Fluid Vorticity omega = dv/dx - du/dy (swirling vortex turbulence from feeding strikes)

    Input:
        frames_rgb: [B, T, 3, H, W] (T >= 1, nominally T=2)
    Output:
        frames_6ch: [B, T, 6, H, W]
        kinematics_summary: [B, 4] lightweight summary tensor [u_mean, v_mean, omega_mean, omega_max]
    """
    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = image_size  # Retained for API compatibility; Sobel filtering is resolution-agnostic

        # Fixed Sobel spatial gradient filters (channel=1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Persistent normalization buffers for physics calculations
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _, H, W = frames_rgb.shape
        device = frames_rgb.device
        dtype = frames_rgb.dtype

        # Unnormalize ImageNet normalization to [0, 1] for physics calculations without CPU-GPU sync
        rgb_flat = frames_rgb.view(B * T, 3, H, W)
        rgb_01 = torch.clamp(rgb_flat * self.rgb_std + self.rgb_mean, 0.0, 1.0)

        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)

        # 1. Grayscale luminance [B, T, 1, H, W]
        gray_flat = 0.299 * rgb_01[:, 0:1] + 0.587 * rgb_01[:, 1:2] + 0.114 * rgb_01[:, 2:3]
        gray_seq = gray_flat.view(B, T, 1, H, W)

        # 2. Spatial gradients Ix, Iy
        ix_flat = F.conv2d(gray_flat, sobel_x, padding=1)
        iy_flat = F.conv2d(gray_flat, sobel_y, padding=1)
        ix_seq = ix_flat.view(B, T, 1, H, W)
        iy_seq = iy_flat.view(B, T, 1, H, W)

        # 3. Temporal differences It
        it_seq = torch.zeros(B, T, 1, H, W, dtype=dtype, device=device)
        if T >= 2:
            it_seq[:, 0] = gray_seq[:, 1] - gray_seq[:, 0]
            it_seq[:, 1:] = gray_seq[:, 1:] - gray_seq[:, :-1]

        # 4. Optical Flow (u, v) via differential gradient formulation:
        grad_sq = ix_seq ** 2 + iy_seq ** 2 + 1e-4
        raw_u = - (ix_seq * it_seq) / grad_sq
        raw_v = - (iy_seq * it_seq) / grad_sq
        u_seq = torch.tanh(raw_u * 2.0)  # [B, T, 1, H, W] in [-1, 1]
        v_seq = torch.tanh(raw_v * 2.0)  # [B, T, 1, H, W] in [-1, 1]

        # 5. Fluid Vorticity omega = dv/dx - du/dy in [-1, 1]
        v_flat = v_seq.view(B * T, 1, H, W)
        u_flat = u_seq.view(B * T, 1, H, W)
        dv_dx = F.conv2d(v_flat, sobel_x, padding=1)
        du_dy = F.conv2d(u_flat, sobel_y, padding=1)
        omega_flat = torch.tanh((dv_dx - du_dy) * 4.0)  # normalized vorticity
        omega_seq = omega_flat.view(B, T, 1, H, W)

        # Assemble finalized 6 channels: [R, G, B, u, v, omega]
        frames_6ch = torch.cat([frames_rgb, u_seq, v_seq, omega_seq], dim=2)  # [B, T, 6, H, W]

        # Lightweight summary tensor [B, 4] (pure kinematics: u, v, omega)
        u_mean = torch.mean(torch.abs(u_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        v_mean = torch.mean(torch.abs(v_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        omega_mean = torch.mean(torch.abs(omega_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        omega_max = torch.amax(torch.abs(omega_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        kinematics_summary = torch.cat([u_mean, v_mean, omega_mean, omega_max], dim=-1)

        return frames_6ch, kinematics_summary

