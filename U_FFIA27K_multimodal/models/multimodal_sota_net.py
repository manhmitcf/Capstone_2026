import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple

from features.motion_kinematics import FishMotionKinematics6Ch
from features.audio_frontend import AudioFrontend
from .video_backbone import ConvNeXtNanoVideoBackbone
from .audio_backbone import AudioMLPBackbone
from .multimodal_fusion import MultimodalTournamentFusion


class MultimodalSOTANet(nn.Module):
    """
    Hierarchical Multimodal Tournament Network with Video Kinematics Tie-Breakers (Unified 384-dim).
    Specifically architected to resolve fish feeding intensity assessment across 4 classes
    (None, Strong, Medium, Weak) via 2-level tournament hierarchy with Configurable Video Kinematics
    Tie-Breakers (B12, B23, B13):

      1. Visual-Kinematic Stream (~2.614M params):
         6-Channel ConvNeXt-Nano (Spatial RGB + Flow (u, v) + Vorticity omega) with Stem Conv2d(6->48, k=4, s=4)
         and Composite LayerNorm temporal aggregation (composite_ln) producing native 384-dim representation.
      2. Acoustic Time-Frequency Stream (~2.173M params):
         High-Resolution TKEO-STFT Audio Frontend (256 kHz, 2049 linear bins, 4,098 params)
         + 2-hidden-layer AudioMLPBackbone (2049 -> 768 -> 512 -> 384, 2,168,448 params).
      3. Pairwise Tournament Fusion (~0.374M params with B23 enabled):
         - Direct Cross-Modal Reliability Gating: [w_V, w_A] = softmax(W[f_V || f_A]).
           f_fused = LayerNorm(w_V * f_V + w_A * f_A) -> Directly fed into Tournament Head.
         - Level 1: Feeding Activity Gating Head (None vs Active Feeding, unified 192-dim).
         - Level 2: 3 Specialized Pairwise Subspace Heads on f_fused
             with Configurable Video Kinematics Referees on f_video (B12, B23, B13).
         - Dynamic Tie-Breaker Intervention (Sigmoid Reparameterization): logit = logit_base + gamma_eff * u_tie * logit_video.
         - Tournament Borda Voting to derive final calibrated multi-class probabilities.
      4. Auxiliary Unimodal Classifier Heads (3,080 params):
         - Video Aux Head: Linear(384, 4)
         - Audio Aux Head: Linear(384, 4)

    Total Parameters:
      - Default (B23 enabled): 5,164,594 (~5.165M) | 121 parameter tensors.
      - All 3 Referees enabled: 5,313,590 (~5.314M) | 135 parameter tensors.
      - Pure Baseline (0 Referees): 5,090,096 (~5.090M) | 114 parameter tensors.
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
        temporal_mode: str = "composite_ln",
        audio_hidden_dims: Tuple[int, ...] = (768, 512),
        tie_breakers: Optional[Any] = None,
        video_drop_path: float = 0.1,
        fusion_type: str = "gating",
        head_type: str = "tournament",
        gate_activation: str = "sigmoid",
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
        self.fusion_type = str(fusion_type).strip().lower()
        self.head_type = str(head_type).strip().lower()
        self.gate_activation = str(gate_activation).strip().lower()

        # 1. Frontends
        self.audio_frontend = audio_frontend if audio_frontend is not None else AudioFrontend()
        self.motion_kinematics = FishMotionKinematics6Ch(image_size=image_size)

        # 2. Backbones (~4.78M)
        self.video_backbone = ConvNeXtNanoVideoBackbone(
            embed_dim=embed_dim,
            in_chans=in_chans,
            num_frames=num_frames,
            drop_path_rate=video_drop_path,
            temporal_mode=temporal_mode,
        )
        self.audio_backbone = AudioMLPBackbone(
            in_features=2049,
            hidden_dims=audio_hidden_dims,
            embed_dim=embed_dim,
            dropout=0.1,
        )

        # 3. Multimodal Fusion Engine (Supports Flat Fusion and Tournament)
        self.fusion = MultimodalTournamentFusion(
            dim=embed_dim,
            classes_num=classes_num,
            tie_breakers=tie_breakers,
            fusion_type=fusion_type,
            head_type=head_type,
            gate_activation=self.gate_activation,
            **kwargs
        )

        # 4. Auxiliary Unimodal Classifier Heads (only when not flat head to guarantee 100% active gradients)
        if self.head_type == "flat":
            self.aux_head_video = None
            self.aux_head_audio = None
        else:
            self.aux_head_video = nn.Linear(embed_dim, classes_num)
            self.aux_head_audio = nn.Linear(embed_dim, classes_num)

    def forward(
        self,
        video_input: torch.Tensor,
        audio_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical End-to-End Forward Pass:
          Raw Inputs:
            - video_input: [B, 2, 3, 224, 224] (RGB multi-frame clip)
            - audio_input: [B, 512000] (2.0s @ 256 kHz raw 1D acoustic waveform)
          Returns:
            - Full prediction dictionary containing tournament voting probabilities,
              calibrated logits, pairwise boundaries, uncertainty, and unimodal logits.
        """
        # Step 1: Preprocessing & Frontend Extraction (with unbatched single-sample guards)
        if video_input.ndim == 4:
            video_input = video_input.unsqueeze(0)
        if video_input.ndim == 5 and video_input.size(2) == 3:
            frames_6ch, _ = self.motion_kinematics(video_input)
        else:
            frames_6ch = video_input

        if audio_input.ndim >= 1 and audio_input.size(-1) > 2049:
            stft_feat = self.audio_frontend(audio_input)
        else:
            stft_feat = audio_input
        if stft_feat.ndim == 1:
            stft_feat = stft_feat.unsqueeze(0)

        # Step 2: Unimodal Spatiotemporal Feature Extraction
        f_video = self.video_backbone(frames_6ch)
        f_audio = self.audio_backbone(stft_feat)

        # Flat Head Fast-Path
        if self.head_type == "flat":
            out = self.fusion(f_video=f_video, f_audio=f_audio)
            out["f_video"] = f_video
            out["f_audio"] = f_audio
            return out

        # Auxiliary Unimodal Logits (for standalone evaluation & multi-objective training)
        logits_video = self.aux_head_video(f_video)
        logits_audio = self.aux_head_audio(f_audio)

        # Step 3: Direct Gated Multimodal Tournament Fusion with Video Tie-Breakers
        fusion_outputs = self.fusion(
            f_video=f_video,
            f_audio=f_audio
        )

        # Step 4: Assemble Comprehensive Output via dict unpacking
        outputs = {
            **fusion_outputs,
            "logits_video": logits_video,
            "logits_audio": logits_audio,
        }
        if "logits" not in outputs and "clipwise_output" in outputs:
            outputs["logits"] = outputs["clipwise_output"]

        # Filter out None values to maintain clean output dictionary
        return {k: v for k, v in outputs.items() if v is not None}
