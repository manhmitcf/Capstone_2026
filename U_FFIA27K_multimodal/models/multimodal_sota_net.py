import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple

from features.motion_kinematics import FishMotionKinematics6Ch, FishMotionKinematics
from features.audio_frontend import AudioFrontend
from .video_backbone import ConvNeXtNanoVideoBackbone
from .audio_backbone import AudioMLPBackbone
from .multimodal_fusion import MultimodalTournamentFusion


class MultimodalSOTANet(nn.Module):
    """
    Multimodal SOTA Network supporting both Multimodal Tournament and Pure Unimodal Video Ablation.

    In Unimodal Video mode (modality='video'):
      - Audio frontend, Audio MLP backbone, Multimodal Tournament Fusion, and Audio Aux Head are bypassed.
      - ConvNeXt-Nano video backbone natively supports dynamic in_chans in {3, 4, 5, 6}:
          * 3 channels: RGB only (~2.613M params)
          * 4 channels: RGB + Fluid Vorticity omega (~2.613M params)
          * 5 channels: RGB + Optical Flow (u, v) (~2.614M params)
          * 6 channels: RGB + Optical Flow (u, v) + Fluid Vorticity omega (~2.615M params)
      - Directly outputs 4-class logits via Linear(384 -> 4).
      - Guarantees 100% active gradient flow with zero dangling parameters.
    """
    model_name: str = "MultimodalSOTANet"

    def __init__(
        self,
        classes_num: int = 4,
        embed_dim: int = 384,
        audio_frontend: Optional[AudioFrontend] = None,
        image_size: int = 224,
        num_frames: int = 2,
        in_chans: int = 6,
        temporal_mode: str = "sum",
        audio_hidden_dims: Tuple[int, ...] = (768, 512),
        tie_breakers: Optional[Any] = None,
        video_drop_path: float = 0.1,
        modality: str = "multimodal",
        **kwargs
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.in_chans = in_chans
        self.temporal_mode = temporal_mode
        self.audio_hidden_dims = audio_hidden_dims
        self.tie_breakers = tie_breakers
        self.video_drop_path = video_drop_path
        self.modality = str(modality).strip().lower() if modality else "multimodal"

        # 1. Frontends & Motion Kinematics
        self.motion_kinematics = FishMotionKinematics(in_chans=in_chans, image_size=image_size)

        # 2. Video Backbone & Classifier Head
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames,
            drop_path_rate=video_drop_path,
            temporal_mode=temporal_mode,
        )
        self.aux_head_video = nn.Linear(embed_dim, classes_num)

        # 3. Audio & Multimodal Fusion (only instantiated when not pure unimodal video)
        if self.modality != "video":
            self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
            self.audio_backbone = AudioMLPBackbone(
                in_features=2049,
                hidden_dims=audio_hidden_dims,
                embed_dim=embed_dim,
                dropout=0.1,
            )
            self.fusion = MultimodalTournamentFusion(
                dim=embed_dim,
                tie_breakers=tie_breakers,
                **kwargs
            )
            self.aux_head_audio = nn.Linear(embed_dim, classes_num)
        else:
            self.audio_frontend = None
            self.audio_backbone = None
            self.fusion = None
            self.aux_head_audio = None

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward Pass:
          In 'video' modality mode:
            - Extracts motion kinematics channels according to in_chans.
            - Processes through ConvNeXt-Nano Video Backbone with temporal_mode aggregation.
            - Projects through classification head.
          In 'multimodal' mode:
            - Executes full hierarchical tournament fusion.
        """
        # --- Pure Unimodal Video Execution Fast-Path ---
        if self.modality == "video":
            if video_input.ndim == 4:
                video_input = video_input.unsqueeze(0)
            if video_input.ndim == 5 and video_input.size(2) == 3:
                frames_ch, _ = self.motion_kinematics(video_input)
            else:
                frames_ch = video_input

            f_video = self.video_backbone(frames_ch)
            logits_video = self.aux_head_video(f_video)
            return {
                "logits": logits_video,
                "clipwise_output": logits_video,
                "logits_video": logits_video,
                "f_video": f_video,
            }

        # --- Multimodal Execution Path ---
        if video_input.ndim == 4:
            video_input = video_input.unsqueeze(0)
        if video_input.ndim == 5 and video_input.size(2) == 3:
            frames_ch, _ = self.motion_kinematics(video_input)
        else:
            frames_ch = video_input

        if audio_input is not None and audio_input.ndim >= 1 and audio_input.size(-1) > 2049:
            stft_feat = self.audio_frontend(audio_input)
        else:
            stft_feat = audio_input if audio_input is not None else torch.zeros(frames_ch.size(0), 2049, device=frames_ch.device)
        if stft_feat.ndim == 1:
            stft_feat = stft_feat.unsqueeze(0)

        f_video = self.video_backbone(frames_ch)
        f_audio = self.audio_backbone(stft_feat)

        logits_video = self.aux_head_video(f_video)
        logits_audio = self.aux_head_audio(f_audio)

        fusion_outputs = self.fusion(
            f_video=f_video,
            f_audio=f_audio
        )

        outputs = {
            **fusion_outputs,
            "logits_video": logits_video,
            "logits_audio": logits_audio,
        }
        if "logits" not in outputs and "clipwise_output" in outputs:
            outputs["logits"] = outputs["clipwise_output"]

        return {k: v for k, v in outputs.items() if v is not None}
