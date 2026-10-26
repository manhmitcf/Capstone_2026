import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from features.audio_frontend import AudioFrontend
from config.train_config import AudioFeaturesConfig
from models.audio_backbone import AudioMLPBackbone, init_layer


class AudioClassifierModel(nn.Module):
    """
    Unimodal Acoustic Classification Model for Feeding Intensity Assessment.
    Pipeline:
      Raw Waveform (B, 512000) @ 256 kHz
        -> AudioFrontend (TKEO-STFT, 2049 linear bins, LayerNorm) [4,098 params]
        -> AudioMLPBackbone (Configurable Depth 1-4, embed_dim=384) [~0.788M to ~3.482M params]
        -> Classifier Head: Linear(384 -> 4) [1,540 params]
        -> Multi-class Logits: [B, 4] for None (0), Strong (1), Medium (2), Weak (3).
    """
    def __init__(
        self,
        audio_config: Optional[AudioFeaturesConfig] = None,
        hidden_dims: Tuple[int, ...] = (768, 512),
        embed_dim: int = 384,
        classes_num: int = 4,
        dropout: float = 0.1,
        use_tkeo: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        if audio_config is None:
            audio_config = AudioFeaturesConfig(use_tkeo=use_tkeo)
        else:
            # Override use_tkeo if explicitly specified
            audio_config.use_tkeo = bool(use_tkeo)

        self.use_tkeo = audio_config.use_tkeo
        self.embed_dim = embed_dim
        self.classes_num = classes_num
        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else ()

        # 1. High-Resolution STFT Frontend (256 kHz, 2049 linear bins)
        self.frontend = AudioFrontend(config=audio_config)

        # 2. Configurable Audio MLP Backbone (Depth 1, 2, 3, or 4)
        self.backbone = AudioMLPBackbone(
            in_features=2049,
            hidden_dims=self.hidden_dims,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # 3. Linear Classification Head
        self.classifier = nn.Linear(embed_dim, classes_num)
        init_layer(self.classifier)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for unimodal acoustic feeding intensity prediction.
        Supports both single-tensor signature forward(audio) and
        multimodal-evaluator signature forward(video, audio).

        Args:
            args: Either (audio_input,) or (video_input, audio_input).
            kwargs: Optional keyword arguments for audio_input/audio_form/audio.

        Returns:
            Dictionary containing:
              - 'logits': [B, 4] multi-class logits
              - 'clipwise_output': [B, 4] alias for evaluator/loss compatibility
              - 'logits_audio': [B, 4] alias for detailed metrics logging
              - 'f_audio': [B, embed_dim] 384-dimensional latent embedding
        """
        if len(args) == 1:
            audio_input = args[0]
        elif len(args) >= 2:
            # Multi-input caller like MultimodalEvaluator or InferenceTimer: 2nd argument is audio
            audio_input = args[1]
        else:
            audio_input = kwargs.get("audio_input", kwargs.get("audio_form", kwargs.get("audio", None)))
            if audio_input is None:
                raise ValueError("AudioClassifierModel requires audio input tensor.")

        # Guard for unbatched 1D input
        if audio_input.ndim == 1:
            audio_input = audio_input.unsqueeze(0)

        # Detect raw acoustic waveform vs pre-extracted STFT vector
        if audio_input.size(-1) > 2049:
            spec = self.frontend(audio_input)
        else:
            spec = audio_input

        # Pass through Audio MLP Backbone
        f_audio = self.backbone(spec)

        # Classify feeding intensity
        logits = self.classifier(f_audio)

        return {
            "logits": logits,
            "clipwise_output": logits,
            "logits_audio": logits,
            "f_audio": f_audio
        }
