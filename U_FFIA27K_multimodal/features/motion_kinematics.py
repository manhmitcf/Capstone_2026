import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FishMotionKinematics6Ch(nn.Module):
    """
    Differentiable Motion Kinematics Extractor for Fish Feeding Intensity Assessment:
      - Channels 0-2 : Spatial RGB appearance (fish clustering, surface foam, feeding pellets)
      - Channels 3-4 : Dense Optical Flow (u, v) representing horizontal and vertical swimming velocities
      - Channel 5    : Fluid Vorticity omega = dv/dx - du/dy (swirling vortex turbulence from feeding strikes)

    Supports dynamic input channel configurations (in_chans in {3, 4, 5, 6}):
      - 3 channels: RGB only
      - 4 channels: RGB + Fluid Vorticity omega
      - 5 channels: RGB + Optical Flow (u, v)
      - 6 channels: RGB + Optical Flow (u, v) + Fluid Vorticity omega

    Input:
        frames_rgb: [B, T, 3, H, W] (T >= 1, nominally T=2)
    Output:
        frames_out: [B, T, in_chans, H, W]
        kinematics_summary: [B, 4] lightweight summary tensor [u_mean, v_mean, omega_mean, omega_max]
    """
    def __init__(self, in_chans: int = 6, image_size: int = 224) -> None:
        super().__init__()
        self.in_chans = in_chans
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

    def forward(self, frames_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _, H, W = frames_rgb.shape
        device = frames_rgb.device
        dtype = frames_rgb.dtype

        if self.in_chans == 3:
            dummy_summary = torch.zeros(B, 4, dtype=dtype, device=device)
            return frames_rgb, dummy_summary

        # Unnormalize ImageNet normalization to [0, 1] for physics calculations
        rgb_flat = frames_rgb.view(B * T, 3, H, W)
        if rgb_flat.min() < 0.0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
            rgb_01 = torch.clamp(rgb_flat * std + mean, 0.0, 1.0)
        else:
            rgb_01 = torch.clamp(rgb_flat, 0.0, 1.0)

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

        # Lightweight summary components
        u_mean = torch.mean(torch.abs(u_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        v_mean = torch.mean(torch.abs(v_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        omega_mean = torch.mean(torch.abs(omega_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)
        omega_max = torch.amax(torch.abs(omega_seq), dim=(1, 2, 3, 4), keepdim=True).view(B, 1)

        if self.in_chans == 4:
            # RGB + Fluid Vorticity omega (4 channels)
            frames_out = torch.cat([frames_rgb, omega_seq], dim=2)
            kinematics_summary = torch.cat([omega_mean, omega_max, torch.zeros(B, 2, dtype=dtype, device=device)], dim=-1)
        elif self.in_chans == 5:
            # RGB + Optical Flow (u, v) (5 channels)
            frames_out = torch.cat([frames_rgb, u_seq, v_seq], dim=2)
            kinematics_summary = torch.cat([u_mean, v_mean, torch.zeros(B, 2, dtype=dtype, device=device)], dim=-1)
        elif self.in_chans == 6:
            # RGB + Optical Flow (u, v) + Fluid Vorticity omega (6 channels)
            frames_out = torch.cat([frames_rgb, u_seq, v_seq, omega_seq], dim=2)
            kinematics_summary = torch.cat([u_mean, v_mean, omega_mean, omega_max], dim=-1)
        else:
            raise ValueError(f"Unsupported in_chans={self.in_chans}. Expected 3, 4, 5, or 6.")

        return frames_out, kinematics_summary


# Backward compatibility alias
FishMotionKinematics = FishMotionKinematics6Ch
