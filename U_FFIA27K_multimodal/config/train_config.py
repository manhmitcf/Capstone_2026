import sys
import json
import logging
from pathlib import Path
from typing import Optional, Literal, Any, Tuple
from pydantic import BaseModel, Field, model_validator

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VALID_CACHE_MODES = ("disk", "ram", "none")


class VideoFeaturesConfig(BaseModel):
    """
    Video frame extraction & spatiotemporal kinematics parameters.
    """
    image_size: int = Field(default=224, description="Target spatial resolution H=W.")
    num_frames: int = Field(default=2, description="Number of uniform frames sampled per video clip (T=2).")
    num_channels: int = Field(default=6, description="6-channel representation (RGB + 2 Differentiable Optical Flow: u, v + Fluid Vorticity: omega).")
    temporal_mode: str = Field(default="composite_ln", description="Temporal aggregation mode: 'composite_ln', 'sum', or 'abs_diff'.")


class AudioFeaturesConfig(BaseModel):
    """
    High-Resolution TKEO-STFT Audio Frontend parameters (256 kHz, 2049 linear bins).
    """
    sample_rate: int = Field(default=256000, description="Audio sampling rate in Hz.")
    window_size: int = Field(default=4096, description="STFT window size in samples.")
    hop_size: int = Field(default=2048, description="STFT hop size in samples.")
    use_tkeo: bool = Field(default=True, description="Enable Teager-Kaiser Energy Operator Adaptive Pre-Emphasis.")
    alpha_max: float = Field(default=0.99, ge=0.0, le=1.0, description="Max pre-emphasis coefficient for TKEO APE.")
    beta: float = Field(default=0.8, ge=0.0, le=1.0, description="Temporal smoothing factor for TKEO APE.")
    use_spectral_aug: bool = Field(default=True, description="Enable 1D Spectral Augmentation (Cutout & Jitter) for STFT.")
    cutout_width: int = Field(default=24, description="Width of contiguous frequency cutout mask in bins.")
    cutout_prob: float = Field(default=0.5, description="Probability of applying frequency cutout per sample.")
    noise_std: float = Field(default=0.02, description="Standard deviation of Gaussian spectral jitter noise.")


class VideoTieBreakersConfig(BaseModel):
    """
    Configuration for 3 Video Kinematics Tie-Breakers on Level 2 pairwise matchups (B12, B23, B13).
    """
    enable_b12: bool = Field(default=False, description="Enable Video Kinematics Tie-Breaker for Weak vs Medium (B12).")
    enable_b23: bool = Field(default=True, description="Enable Video Kinematics Tie-Breaker for Medium vs Strong (B23).")
    enable_b13: bool = Field(default=False, description="Enable Video Kinematics Tie-Breaker for Weak vs Strong (B13).")

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            for key, flat in [("b12", "enable_b12"), ("b23", "enable_b23"), ("b13", "enable_b13")]:
                # 1. Direct flat key
                if flat in d:
                    v = d[flat]
                # 2. Short key
                elif key in d:
                    v = d.pop(key)
                # 3. Split video keys
                else:
                    v_vid = d.get(f"{flat}_video", d.get(f"{key}_video", None))
                    if v_vid is not None:
                        v = v_vid
                    else:
                        continue

                # Parse value into boolean
                if isinstance(v, dict):
                    if "enable_video" in v:
                        res = bool(v["enable_video"])
                    elif "enable" in v:
                        res = bool(v["enable"])
                    elif "video" in v:
                        res = bool(v["video"])
                    else:
                        res = True
                elif hasattr(v, "enable_video"):
                    res = bool(v.enable_video)
                elif hasattr(v, "enable"):
                    res = bool(v.enable)
                else:
                    res = bool(v)

                d[flat] = res
            return d
        return data

    @property
    def b12(self) -> bool:
        return self.enable_b12

    @property
    def b23(self) -> bool:
        return self.enable_b23

    @property
    def b13(self) -> bool:
        return self.enable_b13


class ModelConfig(BaseModel):
    """
    Configuration for Multimodal Tournament Model with Video Kinematics Tie-Breakers (~5.16M parameters).
    Video: ConvNeXt-Nano (6-ch Kinematics, composite_ln) ~2.614M
    Audio: TKEO-STFT-MLP (2049 -> 768 -> 512 -> 384) ~2.168M
    Audio Frontend: TKEO-STFT LayerNorm ~0.004M
    Fusion: Direct Gated Tournament Decision Head with Video Referees ~0.374M (B23)
    Auxiliary Heads: Video + Audio Aux Heads ~0.003M
    """
    backbone: str = Field(
        default="MultimodalSOTANet",
        description="Model architecture name exported by U_FFIA27K_multimodal.models."
    )
    embed_dim: int = Field(default=384, description="Common multimodal embedding dimension.")
    audio_hidden_dims: Tuple[int, ...] = Field(
        default=(768, 512),
        description="Hidden layer dimensions for AudioMLPBackbone."
    )
    classes_num: int = Field(default=4, description="Number of output feeding intensity classes (None, Strong, Medium, Weak).")
    video_drop_path: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Stochastic Depth / DropPath rate for ConvNeXt-Nano video backbone."
    )
    tie_breakers: VideoTieBreakersConfig = Field(
        default_factory=VideoTieBreakersConfig,
        description="Pairwise Video Kinematics Referee configurations for B12, B23, B13."
    )


class SplitterConfig(BaseModel):
    """
    Configuration parameters for dataset splitting.
    """
    dataset_path: str = Field(
        default='/marimo/Fish_Feeding_Intensity_Dataset',
        description="Absolute path to the raw dataset directory."
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for split reproducibility."
    )
    test_sample_per_class: int = Field(
        default=700,
        gt=0,
        description="Number of samples per class designated for test and validation subsets."
    )
    save_results: bool = Field(
        default=True,
        description="Whether to save the splits output results to CSV/JSON files."
    )
    include_video: bool = Field(
        default=True,
        description="Whether to scan video paths alongside audio paths."
    )
    split_strategy: str = Field(
        default="random_sample",
        description="Dataset split strategy: 'random_sample', 'time_series', or 'group_random'."
    )
    evaluation_mode: str = Field(
        default="holdout",
        description="Evaluation mode: 'holdout' or 'cross_validation'."
    )
    num_folds: int = Field(
        default=5,
        gt=1,
        description="Number of outer folds for cross-validation."
    )
    fold_index: Optional[int] = Field(
        default=None,
        description="Current outer fold index for cross-validation."
    )
    cv_val_ratio: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Validation ratio split from the non-test development folds during cross-validation."
    )


class SystemConfig(BaseModel):
    """
    Runtime system, caching, device workers, and storage configuration.
    """
    seed: int = Field(default=42, ge=0, description="Master random seed.")
    cache_mode: str = Field(default="ram", description="Caching mode: 'ram', 'disk', or 'none'.")
    dataloader_workers: int = Field(default=8, description="Number of worker processes for DataLoader (Fixed to 8).")
    prefetch_factor: Optional[int] = Field(default=2, description="Number of batches loaded in advance.")
    ckpt_dir: str = Field(default="checkpoint", description="Directory path to save checkpoints.")


class TrainingConfig(BaseModel):
    """
    Optimization hyperparameters and scheduler configuration.
    """
    epochs: int = Field(default=200, gt=0, description="Total number of training epochs.")
    batch_size: int = Field(default=32, gt=0, description="Training batch size.")
    learning_rate: float = Field(default=1e-3, gt=0, description="Initial learning rate.")
    min_lr: float = Field(default=1e-5, ge=0.0, description="Minimum learning rate for CosineAnnealingLR.")
    weight_decay: float = Field(default=0.05, ge=0, description="Weight decay factor for AdamW.")
    lr_scheduler: Literal["cosine", "constant", "fixed"] = Field(
        default="constant",
        description="Learning rate schedule: 'cosine' (CosineAnnealingLR) or 'constant' (fixed LR across all epochs)."
    )


class EvaluationConfig(BaseModel):
    """
    Validation evaluation, metric monitoring, and early stopping configuration.
    """
    monitor: str = Field(default="val_acc", description="Metric to monitor for best checkpoint: 'val_acc' (or 'accuracy'), 'qwk', or 'both' (dual-track).")
    mode: Literal["min", "max"] = Field(default="max", description="Optimization direction for monitored metric.")
    early_stopping: bool = Field(default=False, description="Enable early stopping mechanism.")
    patience: int = Field(default=40, ge=1, description="Early stopping patience in epochs.")
    min_delta: float = Field(default=0.0, ge=0.0, description="Minimum change threshold in monitored metric.")


class LossConfig(BaseModel):
    """
    Hierarchical Tournament Loss for Level 1 Activity Gate + Level 2 Pairwise Boundaries.
    """
    loss_type: str = Field(default="pairwise_tournament", description="Loss function: 'pairwise_tournament' or 'clip_ce'.")
    weight_act: float = Field(default=0.5, ge=0.0, description="Weight for Level-1 Activity Gate BCE loss.")
    weight_pairwise: float = Field(default=0.5, ge=0.0, description="Weight for Level-2 Pairwise Boundaries loss.")
    weight_ce: float = Field(default=1.0, ge=0.0, description="Weight for Multi-class CE on Tournament Logits.")
    aux_loss_weight: float = Field(default=0.3, ge=0.0, description="Weight for auxiliary unimodal backbone heads.")


class TrainConfig(BaseModel):
    """
    Master configuration schema for multimodal model training, organized into logical clusters:
    system, training, evaluation, loss, model, video_features, audio_features, dataset_splitter.
    """
    system: SystemConfig = Field(default_factory=SystemConfig, description="System runtime, device, and caching settings.")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Optimization and training hyperparameters.")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Validation monitoring and early stopping settings.")
    loss: LossConfig = Field(default_factory=LossConfig, description="Loss function and penalty weights.")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model architecture parameters.")
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig, description="Video preprocessing configuration.")
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig, description="Audio preprocessing configuration.")
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="Dataset splitting settings.")

    @model_validator(mode="before")
    @classmethod
    def _reorganize_clusters(cls, data: Any) -> Any:
        """
        Accepts both clustered JSON dicts and flat dicts, automatically routing
        flat keys into their corresponding logical sub-config clusters.
        """
        if not isinstance(data, dict):
            return data
        d = dict(data)

        # 1. System cluster
        sys_keys = ["seed", "cache_mode", "dataloader_workers", "prefetch_factor", "ckpt_dir"]
        sys_dict = dict(d.get("system", {})) if isinstance(d.get("system"), dict) else {}
        for k in sys_keys:
            if k in d:
                sys_dict[k] = d.pop(k)
        if sys_dict:
            d["system"] = sys_dict

        # 2. Training cluster
        train_keys = ["epochs", "batch_size", "learning_rate", "min_lr", "weight_decay", "lr_scheduler"]
        train_dict = dict(d.get("training", {})) if isinstance(d.get("training"), dict) else {}
        for k in train_keys:
            if k in d:
                train_dict[k] = d.pop(k)
        if train_dict:
            d["training"] = train_dict

        # 3. Evaluation cluster
        eval_keys = ["monitor", "mode", "early_stopping", "patience", "min_delta", "delta"]
        eval_dict = dict(d.get("evaluation", {})) if isinstance(d.get("evaluation"), dict) else {}
        for k in eval_keys:
            if k in d:
                v = d.pop(k)
                if k == "delta":
                    eval_dict.setdefault("min_delta", v)
                else:
                    eval_dict[k] = v
        if eval_dict:
            d["evaluation"] = eval_dict

        # 4. Loss cluster
        loss_keys = ["loss_type", "weight_act", "weight_pairwise", "weight_ce", "aux_loss_weight"]
        loss_dict = dict(d.get("loss", {})) if isinstance(d.get("loss"), dict) else {}
        for k in loss_keys:
            if k in d:
                loss_dict[k] = d.pop(k)
        if loss_dict:
            d["loss"] = loss_dict

        return d

    # -------------------------------------------------------------------------
    # Backward-compatible property getters & setters for seamless flat access
    # -------------------------------------------------------------------------
    @property
    def seed(self) -> int: return self.system.seed
    @seed.setter
    def seed(self, val: int) -> None: self.system.seed = val

    @property
    def cache_mode(self) -> str: return self.system.cache_mode
    @cache_mode.setter
    def cache_mode(self, val: str) -> None: self.system.cache_mode = val

    @property
    def dataloader_workers(self) -> int: return self.system.dataloader_workers
    @dataloader_workers.setter
    def dataloader_workers(self, val: int) -> None: self.system.dataloader_workers = val

    @property
    def prefetch_factor(self) -> Optional[int]: return self.system.prefetch_factor
    @prefetch_factor.setter
    def prefetch_factor(self, val: Optional[int]) -> None: self.system.prefetch_factor = val

    @property
    def ckpt_dir(self) -> str: return self.system.ckpt_dir
    @ckpt_dir.setter
    def ckpt_dir(self, val: str) -> None: self.system.ckpt_dir = val

    @property
    def epochs(self) -> int: return self.training.epochs
    @epochs.setter
    def epochs(self, val: int) -> None: self.training.epochs = val

    @property
    def batch_size(self) -> int: return self.training.batch_size
    @batch_size.setter
    def batch_size(self, val: int) -> None: self.training.batch_size = val

    @property
    def learning_rate(self) -> float: return self.training.learning_rate
    @learning_rate.setter
    def learning_rate(self, val: float) -> None: self.training.learning_rate = val

    @property
    def min_lr(self) -> float: return self.training.min_lr
    @min_lr.setter
    def min_lr(self, val: float) -> None: self.training.min_lr = val

    @property
    def lr_scheduler(self) -> str: return self.training.lr_scheduler
    @lr_scheduler.setter
    def lr_scheduler(self, val: str) -> None: self.training.lr_scheduler = val

    @property
    def weight_decay(self) -> float: return self.training.weight_decay
    @weight_decay.setter
    def weight_decay(self, val: float) -> None: self.training.weight_decay = val

    @property
    def monitor(self) -> str: return self.evaluation.monitor
    @monitor.setter
    def monitor(self, val: str) -> None: self.evaluation.monitor = val

    @property
    def mode(self) -> Literal["min", "max"]: return self.evaluation.mode
    @mode.setter
    def mode(self, val: Literal["min", "max"]) -> None: self.evaluation.mode = val

    @property
    def early_stopping(self) -> bool: return self.evaluation.early_stopping
    @early_stopping.setter
    def early_stopping(self, val: bool) -> None: self.evaluation.early_stopping = val

    @property
    def patience(self) -> int: return self.evaluation.patience
    @patience.setter
    def patience(self, val: int) -> None: self.evaluation.patience = val

    @property
    def min_delta(self) -> float: return self.evaluation.min_delta
    @min_delta.setter
    def min_delta(self, val: float) -> None: self.evaluation.min_delta = val

    @property
    def delta(self) -> float: return self.evaluation.min_delta
    @delta.setter
    def delta(self, val: float) -> None: self.evaluation.min_delta = val

    @property
    def loss_type(self) -> str: return self.loss.loss_type
    @loss_type.setter
    def loss_type(self, val: str) -> None: self.loss.loss_type = val

    @property
    def weight_act(self) -> float: return self.loss.weight_act
    @weight_act.setter
    def weight_act(self, val: float) -> None: self.loss.weight_act = val

    @property
    def weight_pairwise(self) -> float: return self.loss.weight_pairwise
    @weight_pairwise.setter
    def weight_pairwise(self, val: float) -> None: self.loss.weight_pairwise = val

    @property
    def weight_ce(self) -> float: return self.loss.weight_ce
    @weight_ce.setter
    def weight_ce(self, val: float) -> None: self.loss.weight_ce = val

    @property
    def aux_loss_weight(self) -> float: return self.loss.aux_loss_weight
    @aux_loss_weight.setter
    def aux_loss_weight(self, val: float) -> None: self.loss.aux_loss_weight = val


    @property
    def num_frames(self) -> int:
        return self.video_features.num_frames

    @num_frames.setter
    def num_frames(self, val: int) -> None:
        self.video_features.num_frames = val

    @property
    def image_size(self) -> int:
        return self.video_features.image_size

    @image_size.setter
    def image_size(self, val: int) -> None:
        self.video_features.image_size = val

    @property
    def sample_rate(self) -> int:
        return self.audio_features.sample_rate

    @sample_rate.setter
    def sample_rate(self, val: int) -> None:
        self.audio_features.sample_rate = val

    @property
    def in_chans(self) -> int:
        return self.video_features.num_channels

    @in_chans.setter
    def in_chans(self, val: int) -> None:
        self.video_features.num_channels = val

    @property
    def temporal_mode(self) -> str:
        return self.video_features.temporal_mode

    @temporal_mode.setter
    def temporal_mode(self, val: str) -> None:
        self.video_features.temporal_mode = val

    @property
    def audio_hidden_dims(self) -> Tuple[int, ...]:
        return self.model.audio_hidden_dims

    @audio_hidden_dims.setter
    def audio_hidden_dims(self, val: Tuple[int, ...]) -> None:
        self.model.audio_hidden_dims = val


    @classmethod
    def from_json(cls, path: str = "config/train_config.json") -> "TrainConfig":
        config_path = Path(path)
        if not config_path.is_file():
            pkg_fallback = Path(__file__).resolve().parent / Path(path).name
            if pkg_fallback.is_file():
                config_path = pkg_fallback
            else:
                root_fallback = Path(__file__).resolve().parent.parent / path
                if root_fallback.is_file():
                    config_path = root_fallback
        logger.info(f"Loading multimodal training configuration from JSON: '{config_path}'")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def load_train_config(path: str = "config/train_config.json") -> TrainConfig:
    """Convenience helper to load and validate a TrainConfig from JSON."""
    return TrainConfig.from_json(path)
