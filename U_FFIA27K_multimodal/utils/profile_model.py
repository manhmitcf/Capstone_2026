import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculate detailed parameter breakdown for MultimodalSOTANet."""
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    video_params = sum(p.numel() for p in model.video_backbone.parameters() if p.requires_grad) if hasattr(model, "video_backbone") else 0
    audio_params = sum(p.numel() for p in model.audio_backbone.parameters() if p.requires_grad) if hasattr(model, "audio_backbone") else 0
    fusion_params = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad) if hasattr(model, "fusion") else 0
    kinematics_params = sum(p.numel() for p in model.motion_kinematics.parameters() if p.requires_grad) if hasattr(model, "motion_kinematics") else 0
    frontend_params = sum(p.numel() for p in model.audio_frontend.parameters() if p.requires_grad) if hasattr(model, "audio_frontend") else 0
    aux_v_params = sum(p.numel() for p in model.aux_head_video.parameters() if p.requires_grad) if hasattr(model, "aux_head_video") else 0
    aux_a_params = sum(p.numel() for p in model.aux_head_audio.parameters() if p.requires_grad) if hasattr(model, "aux_head_audio") else 0

    breakdown = {
        "video_backbone": video_params,
        "audio_backbone": audio_params,
        "fusion": fusion_params,
        "kinematics": kinematics_params,
        "audio_frontend": frontend_params,
        "aux_head_video": aux_v_params,
        "aux_head_audio": aux_a_params,
        "core_total": video_params + audio_params + fusion_params,
        "total": total_trainable,
        "total_million": total_trainable / 1e6
    }
    return breakdown


def measure_flops(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    num_frames: int = 2
) -> float:
    """Measure total FLOPs for 1 sample inference using native PyTorch FlopCounterMode."""
    if FlopCounterMode is None:
        return 0.0

    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    model.eval()
    dummy_video = torch.randn(1, num_frames, 3, 224, 224, device=device)
    dummy_audio = torch.randn(1, 512000, device=device)

    with FlopCounterMode(display=False) as flop_counter:
        with torch.no_grad():
            _ = model(dummy_video, dummy_audio)

    total_flops = flop_counter.get_total_flops()
    return total_flops / 1e9  # in GFLOPs


def measure_latency(
    model: nn.Module,
    device: Union[str, torch.device] = "cpu",
    warmup: int = 5,
    iterations: int = 20,
    num_frames: int = 2
) -> float:
    """Measure average inference latency in milliseconds."""
    model.to(device)
    model.eval()

    dummy_video = torch.randn(1, num_frames, 3, 224, 224, device=device)
    dummy_audio = torch.randn(1, 512000, device=device)

    is_cuda = (device.type == "cuda") if isinstance(device, torch.device) else str(device).startswith("cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_video, dummy_audio)

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_video, dummy_audio)
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / iterations) * 1000.0
    return avg_latency_ms
