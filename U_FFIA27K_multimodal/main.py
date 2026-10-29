import os
import sys
import copy
import csv
import re
from pathlib import Path
from datetime import datetime
import zipfile
from typing import Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch

from config import ArtifactUploadConfig, TrainConfig
from dataset import FishMultimodalDataLoader
from features.audio_frontend import AudioFrontend
from models import MultimodalSOTANet
from tasks import MultimodalTrainer
from utils import (
    count_parameters,
    measure_flops,
    seed_everything,
    PairwiseTournamentLoss,
    ClipCELoss,
)

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "MultimodalSOTANet": MultimodalSOTANet,
}


def validate_model_config(config: TrainConfig) -> None:
    backbone_name = config.model.backbone
    if backbone_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown multimodal model '{backbone_name}'. Available: {available}.")


def build_model(config: TrainConfig, seed: Optional[int] = None) -> torch.nn.Module:
    validate_model_config(config)
    model_cls = MODEL_REGISTRY[config.model.backbone]
    frontend = AudioFrontend(config.audio_features)

    active_seed = seed if seed is not None else int(getattr(config, "seed", getattr(config.dataset_splitter, "seed", 42)))
    seed_everything(active_seed)

    in_chans = getattr(config, "in_chans", getattr(config.video_features, "num_channels", 6))

    head_type = getattr(config, "head_type", getattr(config.model, "head_type", "flat"))
    fusion_type = getattr(config, "fusion_type", getattr(config.model, "fusion_type", "gating"))
    gate_activation = getattr(config.model, "gate_activation", getattr(config, "gate_activation", "sigmoid"))
    classifier_dropout = getattr(config.model, "classifier_dropout", getattr(config, "classifier_dropout", 0.1))
    tb_cfg = getattr(config.model, "tie_breakers", None)

    if head_type == "flat":
        gate_info = f" (Gate: {gate_activation})" if fusion_type == "gating" else ""
        drop_info = f" [Dropout: {classifier_dropout}]"
        logger.info(
            f"Multimodal Fusion Engine: Flat Head [{fusion_type.upper()}]{gate_info}{drop_info} -> {config.model.classes_num} classes "
            f"(2-Level Tournament & Tie-Breakers bypassed)."
        )
    else:
        if tb_cfg is not None:
            if hasattr(tb_cfg, "enable_b12"):
                b12_str = f"Video={tb_cfg.enable_b12}"
                b23_str = f"Video={tb_cfg.enable_b23}"
                b13_str = f"Video={tb_cfg.enable_b13}"
            elif hasattr(tb_cfg, "b12"):
                b12_str = f"Video={getattr(tb_cfg.b12, 'enable_video', tb_cfg.b12)}"
                b23_str = f"Video={getattr(tb_cfg.b23, 'enable_video', tb_cfg.b23)}"
                b13_str = f"Video={getattr(tb_cfg.b13, 'enable_video', tb_cfg.b13)}"
            elif isinstance(tb_cfg, dict):
                b12_val = tb_cfg.get("enable_b12", tb_cfg.get("b12", True))
                b23_val = tb_cfg.get("enable_b23", tb_cfg.get("b23", True))
                b13_val = tb_cfg.get("enable_b13", tb_cfg.get("b13", True))
                b12_str = f"{b12_val}"
                b23_str = f"{b23_val}"
                b13_str = f"{b13_val}"
            else:
                b12_str = b23_str = b13_str = "Default"
        else:
            b12_str = b23_str = b13_str = "All True"

        logger.info(
            f"Video Tie-Breakers configuration: "
            f"B12 (Weak vs Med)=[{b12_str}], "
            f"B23 (Med vs Strong)=[{b23_str}], "
            f"B13 (Weak vs Strong)=[{b13_str}]"
        )

    return model_cls(
        classes_num=config.model.classes_num,
        embed_dim=config.model.embed_dim,
        audio_frontend=frontend,
        image_size=config.image_size,
        num_frames=config.num_frames,
        in_chans=in_chans,
        temporal_mode=getattr(config.video_features, "temporal_mode", "composite_ln"),
        audio_hidden_dims=getattr(config.model, "audio_hidden_dims", (768, 512)),
        tie_breakers=tb_cfg,
        video_drop_path=getattr(config.model, "video_drop_path", 0.1),
        fusion_type=fusion_type,
        head_type=head_type,
        gate_activation=gate_activation,
        classifier_dropout=classifier_dropout,
    )


def verify_model_dry_run(model: torch.nn.Module, config: TrainConfig, device: torch.device) -> None:
    """
    Fast pre-flight validation of the complete multimodal model pipeline on the target device
    BEFORE loading the dataset into RAM or starting training.

    Validates:
      1. Model parameter audit (~5.16M params).
      2. Device placement of all submodules, weights, and buffer filters.
      3. Forward pass with multi-frame 6-ch kinematics extraction and 2049 STFT bins.
      4. Backward pass & gradient propagation through all trainable parameters.
      5. Evaluation mode forward pass with torch.no_grad().

    If any check fails, logs the error and aborts immediately,
    saving users from waiting through minutes of dataset RAM preloading.
    """
    logger.info("==================================================")
    logger.info("STARTING PRE-FLIGHT DRY-RUN VERIFICATION (Fast Fail Check)...")
    logger.info(f"Target Device: {device}")

    try:
        # 1. Parameter audit
        stats = count_parameters(model)
        is_flat = getattr(model, "head_type", "tournament") == "flat"
        if is_flat:
            fusion_type_str = getattr(model, "fusion_type", getattr(model.fusion, "fusion_type", "gating")).upper()
            logger.info(f"  - Video Backbone (ConvNeXt-Nano 6-ch, composite_ln): {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
            logger.info(f"  - Audio Backbone (TKEO-STFT-MLP [768, 512] -> 384) : {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
            logger.info(f"  - Flat Fusion Layer ({fusion_type_str} -> 4 classes)    : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
            logger.info(f"  * Total Architecture Parameters:                     {stats['core_total']:,} ({stats['core_total']/1e6:.3f} M)")
            logger.info(f"  * Total Trainable Parameters:                        {stats['total']:,} ({stats['total_million']:.3f} M)")
        else:
            logger.info(f"  - Video Backbone (ConvNeXt-Nano 6-ch, composite_ln): {stats['video_backbone']:,} ({stats['video_backbone']/1e6:.3f} M)")
            logger.info(f"  - Audio Backbone (TKEO-STFT-MLP [768, 512] -> 384) : {stats['audio_backbone']:,} ({stats['audio_backbone']/1e6:.3f} M)")
            logger.info(f"  - Tournament Fusion (Direct Gated, 192-ch Heads)   : {stats['fusion']:,} ({stats['fusion']/1e6:.3f} M)")
            logger.info(f"  * Total Architecture Parameters:                     {stats['core_total']:,} ({stats['core_total']/1e6:.3f} M)")
            logger.info(f"  * Total Trainable Parameters:                        {stats['total']:,} ({stats['total_million']:.3f} M) [Limit 5M dropped]")

        try:
            gflops = measure_flops(model, device=device.type, num_frames=getattr(config, "num_frames", 2))
            logger.info(f"  * Inference Complexity (FLOPs):                      {gflops:.4f} GFLOPs")
        except Exception as flop_err:
            logger.warning(f"  * FLOPs profiling skipped: {flop_err}")

        # 2. Test forward pass with dummy tensors
        model.train()
        dummy_video = torch.randn(
            4, config.num_frames, 3, config.image_size, config.image_size,
            device=device, dtype=torch.float32
        )
        audio_length = config.audio_features.sample_rate * 2  # 2 seconds
        dummy_audio = torch.randn(4, audio_length, device=device, dtype=torch.float32)
        dummy_targets = torch.eye(4, device=device, dtype=torch.float32)

        out = model(dummy_video, dummy_audio)
        if "clipwise_output" not in out:
            raise KeyError("Model output missing 'clipwise_output' key.")
        if out["clipwise_output"].shape != (4, 4):
            raise ValueError(f"Expected output shape (4, 4), got {out['clipwise_output'].shape}")

        # 3. Test backward pass & gradient flow
        head_type = getattr(config, "head_type", getattr(config.model, "head_type", "flat"))
        loss_type = getattr(config, "loss_type", "ce")
        if head_type == "flat" or loss_type not in ("pairwise_tournament",):
            loss_fn = ClipCELoss()
            loss = loss_fn(out, {"target": dummy_targets})
        else:
            loss_fn = PairwiseTournamentLoss(
                weight_act=getattr(config, "weight_act", 0.5),
                weight_pairwise=getattr(config, "weight_pairwise", 0.5),
                weight_ce=getattr(config, "weight_ce", 1.0),
                aux_loss_weight=getattr(config, "aux_loss_weight", 0.3),
            ).to(device)
            loss = loss_fn(out, {"target": dummy_targets}, epoch=1)
        loss.backward()

        trainable_with_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        if trainable_with_grads != total_trainable:
            raise RuntimeError(f"Gradient flow broken: only {trainable_with_grads}/{total_trainable} parameters received gradients.")

        model.zero_grad(set_to_none=True)

        # 4. Test evaluation mode forward pass
        model.eval()
        with torch.no_grad():
            _ = model(dummy_video, dummy_audio)

        # 5. Clean up temporary tensors and GPU cache
        del dummy_video, dummy_audio, dummy_targets, out, loss
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(">>> [PASS] PRE-FLIGHT DRY-RUN COMPLETED SUCCESSFULLY!")
        logger.info(f">>> All {total_trainable} parameters, device placement, and gradients 100% verified.")
        logger.info(">>> Proceeding to Dataset RAM Preloading and Full Training Pipeline...")
        logger.info("==================================================")
    except Exception as exc:
        logger.error("==================================================")
        logger.error(f">>> [FATAL ERROR IN PRE-FLIGHT DRY-RUN]: {exc}")
        logger.error(">>> Aborting before loading dataset into RAM to avoid wasting time and resources.")
        logger.error("==================================================")
        raise exc


def model_cv_dir(base_ckpt_dir: str, model_name: str) -> str:
    base_dir = base_ckpt_dir if base_ckpt_dir else "checkpoint"
    if Path(base_dir).name == model_name:
        return base_dir
    return os.path.join(base_dir, model_name)



def read_single_summary_row(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Summary file '{summary_path}' is empty.")
    return rows[-1]


def generate_cv_summary_report(model_dir: Path, num_folds: int) -> Path:
    fold_summaries = []
    for fold_idx in range(num_folds):
        fold_summary_path = model_dir / f"fold_{fold_idx}" / "summary.csv"
        if not fold_summary_path.is_file():
            raise FileNotFoundError(f"Missing summary file for fold {fold_idx}: '{fold_summary_path}'.")
        row = read_single_summary_row(fold_summary_path)
        row["Fold"] = str(fold_idx)
        fold_summaries.append(row)

    metric_keys = [k for k in fold_summaries[0].keys() if k != "Fold"]
    mean_row = {"Fold": "mean"}
    std_row = {"Fold": "std"}

    for key in metric_keys:
        values = []
        for summary in fold_summaries:
            try:
                values.append(float(summary[key]))
            except ValueError:
                pass
        if values:
            mean_row[key] = f"{float(sum(values) / len(values)):.6f}"
            variance = sum((x - float(mean_row[key])) ** 2 for x in values) / len(values)
            std_row[key] = f"{float(variance ** 0.5):.6f}"
        else:
            mean_row[key] = "N/A"
            std_row[key] = "N/A"

    cv_summary_path = model_dir / "cv_summary.csv"
    fieldnames = ["Fold"] + metric_keys
    with cv_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_summaries:
            writer.writerow(row)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    logger.info(f"Cross-validation summary successfully saved to: '{cv_summary_path}'")
    return cv_summary_path


def build_artifact_filename(config: TrainConfig, timestamp: str, suffix: str = ".zip") -> str:
    case_name = getattr(config, "case_name", getattr(config.model, "case_name", None))
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    if case_name:
        return f"{case_name}_{timestamp}{safe_suffix}"
    model_name = config.model.backbone
    dataset_name = Path(config.dataset_splitter.dataset_path).name or "dataset"
    split_strategy = config.dataset_splitter.split_strategy
    eval_mode = config.dataset_splitter.evaluation_mode
    return f"{model_name}_{dataset_name}_{split_strategy}_{eval_mode}_{timestamp}{safe_suffix}"


def zip_directory(source_dir: str, output_path: str) -> str:
    source_path = Path(source_dir).resolve()
    target_path = Path(output_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory to zip does not exist: '{source_path}'")

    logger.info(f"Creating artifact zip from '{source_path}' to '{target_path}'...")
    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in source_path.rglob("*"):
            if not file_path.is_file():
                continue
            resolved_file = file_path.resolve()
            if resolved_file == target_path or ".git" in resolved_file.parts or resolved_file.suffix == ".zip":
                continue
            zip_file.write(resolved_file, arcname=resolved_file.relative_to(source_path))

    logger.info(f"Successfully created artifact zip: '{target_path}'")
    return str(target_path)


def discover_hf_token() -> Optional[str]:
    """Find Hugging Face token across environment variables and setup scripts."""
    token = os.environ.get("HF_TOKEN")
    if token:
        token = token.strip().lstrip('\ufeff')
        if token:
            return token

    # Search for token in dedicated local token.txt files (gitignored)
    direct_token_files = [
        Path("token.txt"),
        Path("../token.txt"),
        Path("/marimo/token.txt"),
        Path("/marimo/Capstone_2026_Fish_Feeding_Intensity/token.txt"),
        Path(__file__).resolve().parent.parent / "token.txt",
    ]
    for dtf in direct_token_files:
        if dtf.is_file():
            try:
                content = dtf.read_text(encoding="utf-8-sig").strip().lstrip('\ufeff')
                match = re.search(r'(hf_[A-Za-z0-9_]+)', content)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass

    try:
        from huggingface_hub import get_token
        hub_token = get_token()
        if hub_token:
            return hub_token.strip().lstrip('\ufeff')
    except Exception:
        pass

    # Search for token strings in setup files
    token_files = [
        Path("cli.txt"),
        Path("/marimo/cli.txt"),
        Path("run_marimo.txt"),
        Path("/marimo/run_marimo.txt"),
        Path(__file__).resolve().parent.parent / "cli.txt",
        Path(__file__).resolve().parent.parent / "run_marimo.txt"
    ]
    for tf in token_files:
        if tf.is_file():
            try:
                content = tf.read_text(encoding="utf-8-sig")
                match = re.search(r'HF_TOKEN\s*=\s*["\'](hf_[A-Za-z0-9_]+)["\']', content)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass
    return None


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, config: TrainConfig) -> None:
    if not upload_config.enabled:
        logger.info("Hugging Face upload is disabled. Skipping upload step.")
        return

    if not upload_config.repo_id:
        raise ValueError("artifact_upload.repo_id must be set when upload is enabled.")

    token = discover_hf_token()
    if not token:
        logger.warning("HF_TOKEN not found. Skipping Hugging Face upload.")
        return

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for artifact upload. Install via pip.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_filename = build_artifact_filename(config, timestamp)
    source_dir = upload_config.source_dir
    if not Path(source_dir).exists():
        # Fallback to current project root
        source_dir = str(Path(__file__).resolve().parent.parent)

    zip_parent = Path(upload_config.zip_path).parent if upload_config.zip_path else Path(source_dir).parent
    output_zip_path = str(zip_parent / artifact_filename)
    artifact_zip = zip_directory(source_dir, output_zip_path)

    target_branch = getattr(upload_config, "branch", None) or "main"
    if upload_config.create_repo:
        try:
            create_repo(
                repo_id=upload_config.repo_id,
                repo_type=upload_config.repo_type,
                token=token,
                exist_ok=True,
            )
        except Exception as exc:
            logger.warning(f"Could not create or verify HF repository '{upload_config.repo_id}': {exc}")

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_branch(
                repo_id=upload_config.repo_id,
                branch=target_branch,
                repo_type=upload_config.repo_type,
                exist_ok=True
            )
            logger.info(f"Verified branch '{target_branch}' on HF repo '{upload_config.repo_id}'.")
        except Exception as exc:
            logger.debug(f"Branch verification note: {exc}")

    logger.info("==================================================")
    logger.info(f"Uploading artifact to Hugging Face repo: '{upload_config.repo_id}'")
    logger.info(f"Repo type:                            '{upload_config.repo_type}'")
    logger.info(f"Branch:                               '{target_branch}'")
    logger.info("==================================================")

    try:
        upload_file(
            path_or_fileobj=artifact_zip,
            path_in_repo=artifact_filename,
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
            revision=target_branch,
        )
        logger.info("Artifact upload to Hugging Face completed successfully!")
    except Exception as exc:
        logger.error(f"Hugging Face artifact upload failed: {exc}")
        logger.info(f"Local training checkpoints remain completely intact at: '{config.ckpt_dir}'")


def run_training_session(
    train_config_path: Optional[str] = None,
    artifact_upload_config_path: Optional[str] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    seed: Optional[int] = None,
    evaluation_mode: Optional[str] = None,
    fold_index: Optional[int] = None,
    lr_scheduler: Optional[str] = None,
    eta_min: Optional[float] = None,
    fusion_type: Optional[str] = None,
    gate_activation: Optional[str] = None,
    classifier_dropout: Optional[float] = None,
    case_name: Optional[str] = None,
    resume: bool = False,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> None:
    pkg_dir = Path(__file__).resolve().parent
    if train_config_path is None:
        train_config_path = str(pkg_dir / "config" / "train_config.json")
    if artifact_upload_config_path is None:
        artifact_upload_config_path = str(pkg_dir / "config" / "artifact_upload_config.json")

    config = TrainConfig.from_json(train_config_path)
    if fusion_type is not None:
        config.fusion_type = fusion_type
    if gate_activation is not None:
        config.gate_activation = gate_activation
    if classifier_dropout is not None:
        config.classifier_dropout = classifier_dropout
    if case_name is not None:
        config.case_name = case_name
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if seed is not None:
        config.seed = seed
        config.dataset_splitter.seed = seed
    if evaluation_mode is not None:
        config.dataset_splitter.evaluation_mode = evaluation_mode
    if fold_index is not None:
        config.dataset_splitter.fold_index = fold_index
    if lr_scheduler is not None:
        config.lr_scheduler = lr_scheduler
    if eta_min is not None:
        config.eta_min = eta_min

    # Lock deterministic master seed across PyTorch, CUDA, NumPy, Random
    active_seed = int(getattr(config, "seed", getattr(config.dataset_splitter, "seed", 42)))
    seed_everything(active_seed)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    eval_mode = config.dataset_splitter.evaluation_mode
    case_name = getattr(config, "case_name", getattr(config.model, "case_name", None))
    model_name = case_name if case_name else config.model.backbone

    # =========================================================================
    # FAST PRE-FLIGHT DRY-RUN (Verify full network before preloading RAM)
    # =========================================================================
    preflight_model = build_model(config, seed=active_seed).to(device)
    verify_model_dry_run(preflight_model, config, device)

    if dry_run:
        logger.info(">>> Dry-run flag detected: Pre-flight check passed. Exiting without training.")
        return

    if eval_mode == "cross_validation":
        del preflight_model
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_folds = config.dataset_splitter.num_folds
        logger.info(f"Starting {num_folds}-Fold Cross-Validation for {model_name}...")
        base_dir = Path(model_cv_dir(config.ckpt_dir, model_name))

        target_folds = (
            [int(config.dataset_splitter.fold_index)]
            if config.dataset_splitter.fold_index is not None
            else list(range(num_folds))
        )

        for fold_idx in target_folds:
            logger.info(f"===== RUNNING FOLD {fold_idx + 1}/{num_folds} =====")
            fold_seed = active_seed + fold_idx
            seed_everything(fold_seed)

            fold_config = copy.deepcopy(config)
            fold_config.dataset_splitter.fold_index = fold_idx

            data_loader = FishMultimodalDataLoader(
                batch_size=fold_config.batch_size,
                dataloader_workers=fold_config.dataloader_workers,
                prefetch_factor=getattr(fold_config, "prefetch_factor", 2),
                cache_mode=fold_config.cache_mode,
                image_size=fold_config.image_size,
                num_frames=fold_config.num_frames,
                sample_rate=fold_config.sample_rate,
                splitter_config=fold_config.dataset_splitter,
            )

            model = build_model(fold_config, seed=fold_seed).to(device)
            stats = count_parameters(model)
            logger.info(f"Fold {fold_idx} Model Parameters: {stats['total']:,} ({stats['total_million']:.3f} M)")

            trainer = MultimodalTrainer(
                model=model,
                train_loader=data_loader.get_data_loader(split='train'),
                val_loader=data_loader.get_data_loader(split='val'),
                test_loader=data_loader.get_data_loader(split='test'),
                config=fold_config,
                device=device,
                train_config_path=train_config_path
            )
            trainer.train(resume=resume)

        if len(target_folds) == num_folds:
            generate_cv_summary_report(base_dir, num_folds)
    else:
        logger.info(f"Starting Holdout Training for {model_name} (Epochs={config.epochs}, Patience={config.patience})...")
        data_loader = FishMultimodalDataLoader(
            batch_size=config.batch_size,
            dataloader_workers=config.dataloader_workers,
            prefetch_factor=getattr(config, "prefetch_factor", 2),
            cache_mode=config.cache_mode,
            image_size=config.image_size,
            num_frames=config.num_frames,
            sample_rate=config.sample_rate,
            splitter_config=config.dataset_splitter,
        )

        model = preflight_model

        trainer = MultimodalTrainer(
            model=model,
            train_loader=data_loader.get_data_loader(split='train'),
            val_loader=data_loader.get_data_loader(split='val'),
            test_loader=data_loader.get_data_loader(split='test'),
            config=config,
            device=device,
            train_config_path=train_config_path
        )
        trainer.train(resume=resume)

    # Upload artifacts to Hugging Face
    upload_config = ArtifactUploadConfig.from_json(artifact_upload_config_path)
    upload_artifact_if_enabled(upload_config, config)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal SOTA Fish Feeding Intensity Assessment")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.json")
    parser.add_argument("--upload-config", type=str, default=None, help="Path to artifact_upload_config.json")
    parser.add_argument("--device", type=str, default=None, help="Target compute device (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-flight check only without training")
    parser.add_argument("--seed", type=int, default=None, help="Master random seed for reproducibility (default: 42 from config)")
    parser.add_argument("--evaluation-mode", type=str, choices=["holdout", "cross_validation"], default=None, help="Override evaluation mode (holdout or cross_validation)")
    parser.add_argument("--fold-index", type=int, default=None, help="Specific CV fold index to run (0 to num_folds - 1)")
    parser.add_argument("--lr-scheduler", type=str, choices=["onecycle", "cosine", "constant", "fixed", "none"], default=None, help="Override learning rate scheduler")
    parser.add_argument("--eta-min", type=float, default=None, help="Override minimum LR (eta_min) for CosineAnnealingLR")
    parser.add_argument("--resume", action="store_true", help="Resume training from last_model.pth if available")
    parser.add_argument("--fusion", "-f", type=str, choices=["gating", "sum", "concat"], default=None, help="Flat multimodal fusion type (gating, sum, or concat)")
    parser.add_argument("--gate-activation", type=str, choices=["sigmoid", "softmax"], default=None, help="Gate activation function ('sigmoid' or 'softmax')")
    parser.add_argument("--classifier-dropout", "--dropout", dest="classifier_dropout", type=float, default=None, help="Dropout probability before flat classifier linear layer")
    parser.add_argument("--case", type=str, choices=["fusion_gating", "fusion_sum", "fusion_concat", "gating", "sum", "concat"], default=None, help="Ablation case name or shorthand")
    parser.add_argument("--epochs", type=int, default=None, help="Override total training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=None, help="Override initial learning rate")
    args = parser.parse_args()

    run_training_session(
        train_config_path=args.config,
        artifact_upload_config_path=args.upload_config,
        dry_run=args.dry_run,
        device_str=args.device,
        seed=args.seed,
        evaluation_mode=args.evaluation_mode,
        fold_index=args.fold_index,
        lr_scheduler=args.lr_scheduler,
        eta_min=args.eta_min,
        resume=args.resume,
        fusion_type=args.fusion,
        gate_activation=args.gate_activation,
        classifier_dropout=args.classifier_dropout,
        case_name=args.case,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()


