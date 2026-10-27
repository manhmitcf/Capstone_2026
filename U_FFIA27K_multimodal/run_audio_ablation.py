#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================================
UNIMODAL ACOUSTIC ABLATION RUNNER: 8 CONFIGURATIONS (Branch: ablations/audio_depth_tkeo)
========================================================================================
Executes full ablation matrix for the Acoustic branch:
  - 2 TKEO Conditions: Disabled (False) vs Enabled (True)
  - 4 Depth Variants:
      * Depth 1: No hidden layers (Linear: 2049 -> 384) [~0.794M params]
      * Depth 2: 1 hidden layer   (Linear: 2049 -> 512 -> 384) [~1.254M params]
      * Depth 3: 2 hidden layers  (Linear: 2049 -> 768 -> 512 -> 384) [~2.174M params - Canonical]
      * Depth 4: 3 hidden layers  (Linear: 2049 -> 1024 -> 768 -> 512 -> 384) [~3.488M params]

Optimization & Hyperparameters:
  - Multi-Class Cross-Entropy Loss (ClipCELoss equivalent)
  - AdamW Optimizer, fixed learning_rate = 1e-4 (constant), weight_decay = 0.05
  - Target Branch on Hugging Face: 'ablations/audio_depth_tkeo'
  - Automatic per-case artifact packaging and Hugging Face upload

Usage:
  python run_audio_ablation.py --dry-run
  python run_audio_ablation.py --epochs 200 --device cuda
  python run_audio_ablation.py --resume
========================================================================================
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure stdout/stderr UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.train_config import TrainConfig
from dataset.dataloader_multimodal import FishMultimodalDataLoader
from models.audio_classifier import AudioClassifierModel
from tasks.audio_train import AudioTrainer
from utils.hf_ablation_uploader import (
    upload_audio_case_to_hf,
    upload_summary_file_to_hf,
    discover_hf_token,
)
from utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AudioAblation")

# 8 Unimodal Acoustic Configurations
AUDIO_ABLATION_CASES: List[Dict[str, Any]] = [
    {
        "case_name": "audio_d1_notkeo",
        "use_tkeo": False,
        "hidden_dims": (),
        "depth": 1,
        "description": "Depth 1 (2049 -> 384), Without TKEO Pre-Emphasis",
    },
    {
        "case_name": "audio_d2_notkeo",
        "use_tkeo": False,
        "hidden_dims": (512,),
        "depth": 2,
        "description": "Depth 2 (2049 -> 512 -> 384), Without TKEO Pre-Emphasis",
    },
    {
        "case_name": "audio_d3_notkeo",
        "use_tkeo": False,
        "hidden_dims": (768, 512),
        "depth": 3,
        "description": "Depth 3 (2049 -> 768 -> 512 -> 384), Without TKEO Pre-Emphasis",
    },
    {
        "case_name": "audio_d4_notkeo",
        "use_tkeo": False,
        "hidden_dims": (1024, 768, 512),
        "depth": 4,
        "description": "Depth 4 (2049 -> 1024 -> 768 -> 512 -> 384), Without TKEO Pre-Emphasis",
    },
    {
        "case_name": "audio_d1_tkeo",
        "use_tkeo": True,
        "hidden_dims": (),
        "depth": 1,
        "description": "Depth 1 (2049 -> 384), With TKEO Pre-Emphasis",
    },
    {
        "case_name": "audio_d2_tkeo",
        "use_tkeo": True,
        "hidden_dims": (512,),
        "depth": 2,
        "description": "Depth 2 (2049 -> 512 -> 384), With TKEO Pre-Emphasis",
    },
    {
        "case_name": "audio_d3_tkeo",
        "use_tkeo": True,
        "hidden_dims": (768, 512),
        "depth": 3,
        "description": "Depth 3 (2049 -> 768 -> 512 -> 384), With TKEO Pre-Emphasis [Proposed Baseline]",
    },
    {
        "case_name": "audio_d4_tkeo",
        "use_tkeo": True,
        "hidden_dims": (1024, 768, 512),
        "depth": 4,
        "description": "Depth 4 (2049 -> 1024 -> 768 -> 512 -> 384), With TKEO Pre-Emphasis",
    },
]


def create_synthetic_dataloaders(batch_size: int = 4, device: torch.device = torch.device("cpu")):
    """Create lightweight synthetic DataLoader for fast pre-flight verification."""
    class SyntheticAblationDataset(torch.utils.data.Dataset):
        def __init__(self, size: int = 8):
            self.size = size
            self.video_data = torch.zeros(size, 2, 3, 224, 224, dtype=torch.float32)
            self.audio_data = torch.randn(size, 512000, dtype=torch.float32)
            self.targets = torch.tensor([i % 4 for i in range(size)], dtype=torch.float32)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "clip_name": f"synthetic_sample_{idx:03d}",
                "video_form": self.video_data[idx],
                "audio_form": self.audio_data[idx],
                "target": self.targets[idx],
            }

    train_ds = SyntheticAblationDataset(size=8)
    val_ds = SyntheticAblationDataset(size=4)
    test_ds = SyntheticAblationDataset(size=4)

    def collate_fn(batch):
        return {
            "clip_name": [item["clip_name"] for item in batch],
            "video_form": torch.stack([item["video_form"] for item in batch]),
            "audio_form": torch.stack([item["audio_form"] for item in batch]),
            "target": torch.stack([item["target"] for item in batch]),
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def update_summary_tables(summary_rows: List[Dict[str, Any]], base_dir: Path) -> Tuple[Path, Path]:
    """Generate both CSV and Markdown comparison summary tables."""
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "audio_ablation_summary.csv"
    md_path = base_dir / "audio_ablation_summary.md"

    fieldnames = [
        "case_name",
        "use_tkeo",
        "depth",
        "hidden_dims",
        "trainable_params",
        "val_acc",
        "val_qwk",
        "test_acc",
        "test_qwk",
        "test_f1",
        "latency_ms",
        "gflops",
        "best_epoch",
        "total_time_seconds",
    ]

    df = pd.DataFrame(summary_rows)
    # Reorder columns that exist
    ordered_cols = [c for c in fieldnames if c in df.columns] + [c for c in df.columns if c not in fieldnames]
    df = df[ordered_cols]
    df.to_csv(csv_path, index=False)

    # Format Markdown Table
    md_lines = [
        "# Unimodal Audio Ablation Study Results",
        "",
        "**Branch**: `ablations/audio_depth_tkeo`  ",
        "**Ablation Space**: 2 TKEO conditions (Disabled / Enabled) × 4 MLP Depths (Depth 1, 2, 3, 4) = 8 Cases  ",
        "**Optimization**: AdamW, Learning Rate = 1e-4 (constant), Cross-Entropy Loss  ",
        "",
        "| Case | TKEO | Depth | Hidden Dims | Trainable Params | Val Acc (%) | Val QWK | Test Acc (%) | Test QWK | Test F1 | Latency (ms) | GFLOPs |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in summary_rows:
        val_acc_pct = f"{row.get('val_acc', 0.0) * 100:.2f}%"
        test_acc_pct = f"{row.get('test_acc', 0.0) * 100:.2f}%"
        val_qwk_str = f"{row.get('val_qwk', 0.0):.4f}"
        test_qwk_str = f"{row.get('test_qwk', 0.0):.4f}"
        test_f1_str = f"{row.get('test_f1', 0.0):.4f}"
        params_str = f"{row.get('trainable_params', 0):,}"
        latency_str = f"{row.get('latency_ms', 0.0):.2f}"
        gflops_str = f"{row.get('gflops', 0.0):.4f}"
        tkeo_str = "Yes" if row.get("use_tkeo", True) else "No"
        depth_val = row.get("depth", "-")
        dims_str = f"`{row.get('hidden_dims', '[]')}`"
        md_lines.append(
            f"| `{row.get('case_name')}` | {tkeo_str} | {depth_val} | {dims_str} | {params_str} | {val_acc_pct} | {val_qwk_str} | {test_acc_pct} | {test_qwk_str} | {test_f1_str} | {latency_str} | {gflops_str} |"
        )

    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path


def run_audio_ablation_suite(
    dry_run: bool = False,
    epochs: Optional[int] = None,
    learning_rate: float = 1e-4,
    device_str: Optional[str] = None,
    resume: bool = False,
    cases_filter: Optional[List[str]] = None,
    hf_upload: bool = True,
    hf_branch: str = "main",
    train_config_path: Optional[str] = None,
) -> None:
    """Execute the full 8-case unimodal acoustic ablation suite."""
    pkg_dir = Path(__file__).resolve().parent
    if train_config_path is None:
        train_config_path = str(pkg_dir / "config" / "train_config.json")

    config = TrainConfig.from_json(train_config_path)

    # Resolve target device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set master random seed
    active_seed = int(getattr(config, "seed", getattr(config.dataset_splitter, "seed", 42)))
    seed_everything(active_seed)

    effective_epochs = 1 if dry_run else (epochs if epochs is not None else config.epochs)
    effective_lr = learning_rate if learning_rate is not None else config.learning_rate

    base_ckpt_dir = pkg_dir / "checkpoint" / "audio_ablations"
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("STARTING UNIMODAL AUDIO ABLATION SUITE (8 CASES)")
    logger.info(f"  - Mode:              {'DRY-RUN (Pre-flight fast check)' if dry_run else 'FULL TRAINING'}")
    logger.info(f"  - Device:            {device}")
    logger.info(f"  - Epochs per case:   {effective_epochs}")
    logger.info(f"  - Learning Rate:     {effective_lr:.2e}")
    logger.info(f"  - Checkpoint Dir:    {base_ckpt_dir}")
    logger.info(f"  - HF Target Branch:  {hf_branch}")
    logger.info("================================================================================")

    # 1. Setup DataLoaders
    if dry_run:
        logger.info(">>> Dry-run active: Initializing synthetic fast DataLoader...")
        train_loader, val_loader, test_loader = create_synthetic_dataloaders(batch_size=4, device=device)
    else:
        dataset_path = getattr(config.dataset_splitter, "dataset_path", "")
        if not os.path.exists(dataset_path):
            logger.error(
                f"Dataset path '{dataset_path}' does not exist on this machine.\n"
                f"Please ensure the dataset is cloned to '{dataset_path}' as specified in cli.txt, "
                f"or run with `--dry-run` to verify architecture and code integrity."
            )
            sys.exit(1)

        logger.info(f"Loading real dataset from '{dataset_path}' (Audio-only RAM caching)...")
        data_loader = FishMultimodalDataLoader(
            batch_size=config.batch_size,
            dataloader_workers=config.dataloader_workers,
            persistent_workers=getattr(config, "persistent_workers", True),
            prefetch_factor=getattr(config, "prefetch_factor", 2),
            cache_mode=config.cache_mode,
            image_size=config.image_size,
            num_frames=config.num_frames,
            sample_rate=config.sample_rate,
            splitter_config=config.dataset_splitter,
            load_video=False,
            load_audio=True,
        )
        train_loader = data_loader.get_data_loader(split="train")
        val_loader = data_loader.get_data_loader(split="val")
        test_loader = data_loader.get_data_loader(split="test")

    # Filter target cases if requested
    target_cases = AUDIO_ABLATION_CASES
    if cases_filter:
        target_cases = [c for c in AUDIO_ABLATION_CASES if c["case_name"] in cases_filter]
        if not target_cases:
            logger.error(f"No matching cases found for filter: {cases_filter}")
            sys.exit(1)

    summary_rows: List[Dict[str, Any]] = []

    # 2. Iterate through each ablation case
    total_cases = len(target_cases)
    for idx, case_cfg in enumerate(target_cases, 1):
        case_name = case_cfg["case_name"]
        use_tkeo = case_cfg["use_tkeo"]
        hidden_dims = case_cfg["hidden_dims"]
        case_dir = base_ckpt_dir / case_name

        logger.info("")
        logger.info("================================================================================")
        logger.info(f"CASE {idx}/{total_cases}: {case_name}")
        logger.info(f"  Description: {case_cfg['description']}")
        logger.info(f"  TKEO:        {use_tkeo}")
        logger.info(f"  Hidden Dims: {hidden_dims} (Depth {len(hidden_dims) + 1})")
        logger.info("================================================================================")

        # Check for resume
        if resume and (case_dir / "best_model.pth").exists():
            rep_json = case_dir / "evaluation_detailed_report.json"
            summ_csv = case_dir / "summary.csv"
            resumed = False
            if rep_json.exists():
                try:
                    with open(rep_json, "r", encoding="utf-8") as f:
                        rep = json.load(f)
                    val_m = rep.get("validation_metrics", {})
                    test_m = rep.get("test_metrics", {})
                    resumed_summary = {
                        "case_name": case_name,
                        "use_tkeo": use_tkeo,
                        "depth": case_cfg["depth"],
                        "hidden_dims": str(list(hidden_dims)),
                        "trainable_params": int(rep.get("trainable_params", 0)),
                        "val_acc": float(val_m.get("accuracy", 0.0)),
                        "val_qwk": float(val_m.get("qwk", 0.0)),
                        "test_acc": float(test_m.get("accuracy", 0.0)),
                        "test_qwk": float(test_m.get("qwk", 0.0)),
                        "test_f1": float(test_m.get("f1_macro", 0.0)),
                        "latency_ms": float(rep.get("latency_ms", 0.0)),
                        "gflops": float(rep.get("gflops", 0.0)),
                        "best_epoch": effective_epochs,
                        "total_time_seconds": float(rep.get("training_duration_s", 0.0)),
                    }
                    logger.info(f"[{case_name}] Found valid evaluation report. Resuming / skipping run...")
                    summary_rows.append(resumed_summary)
                    resumed = True
                    continue
                except Exception as exc:
                    logger.warning(f"Could not parse evaluation_detailed_report.json for {case_name}: {exc}")
            if not resumed and summ_csv.exists():
                try:
                    prev_summary = pd.read_csv(summ_csv)
                    if len(prev_summary) > 0:
                        row = prev_summary.iloc[0].to_dict()
                        row["case_name"] = case_name
                        row["use_tkeo"] = use_tkeo
                        row["depth"] = case_cfg["depth"]
                        row["hidden_dims"] = str(list(hidden_dims))
                        summary_rows.append(row)
                        continue
                except Exception as exc:
                    logger.warning(f"Could not read previous summary for {case_name}: {exc}. Rerunning.")

        # Re-seed to guarantee exact reproducibility for each case
        seed_everything(active_seed + idx)

        # Build Unimodal Acoustic Model
        model = AudioClassifierModel(
            audio_config=config.audio_features,
            hidden_dims=hidden_dims,
            embed_dim=config.model.embed_dim,
            classes_num=config.model.classes_num,
            use_tkeo=use_tkeo,
        ).to(device)

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        logger.info(f"[{case_name}] Trainable Parameters: {trainable_params:,} (Backbone: {backbone_params:,})")

        # Fast Gradient Verification Check
        dummy_audio = torch.randn(2, 512000, device=device)
        dummy_target = torch.tensor([0, 1], dtype=torch.long, device=device)
        model.train()
        outputs = model(dummy_audio)
        logits = outputs["logits"]
        loss = nn.CrossEntropyLoss()(logits, dummy_target)
        loss.backward()

        zero_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
        if zero_grad_params:
            raise RuntimeError(f"[{case_name}] Gradient check failed for parameters: {zero_grad_params}")
        logger.info(f"[{case_name}] Pre-flight gradient check passed: 100% of parameter tensors receive gradients.")
        model.zero_grad()

        # Build Dedicated Audio Trainer
        trainer = AudioTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=effective_epochs,
            learning_rate=effective_lr,
            weight_decay=config.weight_decay,
            ckpt_dir=str(base_ckpt_dir),
            case_name=case_name,
            is_dry_run=dry_run,
        )

        # Execute Training & Evaluation
        case_summary = trainer.train()
        summary_rows.append(case_summary)

        # Update consolidated summary tables
        csv_path, md_path = update_summary_tables(summary_rows, base_ckpt_dir)

        # Upload results of this case immediately to Hugging Face
        if hf_upload and not dry_run:
            logger.info(f"[{case_name}] Pushing case output directory to Hugging Face...")
            upload_audio_case_to_hf(
                case_name=case_name,
                case_dir=str(case_dir),
                branch_name=hf_branch,
            )
            # Also upload the updated consolidated summary files
            upload_summary_file_to_hf(str(csv_path), branch_name=hf_branch)
            upload_summary_file_to_hf(str(md_path), branch_name=hf_branch)

        # Explicit garbage collection to release GPU VRAM and CPU memory between runs
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # 3. Print Final Summary Table
    logger.info("")
    logger.info("================================================================================")
    logger.info("ALL 8 UNIMODAL ACOUSTIC ABLATION RUNS COMPLETED SUCCESSFULLY!")
    logger.info("================================================================================")
    summary_md_file = base_ckpt_dir / "audio_ablation_summary.md"
    if summary_md_file.exists():
        print("\n" + summary_md_file.read_text(encoding="utf-8"))

    # Upload final consolidated summary files if dry_run == False
    if hf_upload and not dry_run:
        csv_path, md_path = update_summary_tables(summary_rows, base_ckpt_dir)
        upload_summary_file_to_hf(str(csv_path), branch_name=hf_branch)
        upload_summary_file_to_hf(str(md_path), branch_name=hf_branch)
        logger.info(f"Final summary tables pushed to Hugging Face branch '{hf_branch}'.")


def main():
    parser = argparse.ArgumentParser(description="Unimodal Acoustic Ablation Suite (8 Cases)")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-flight verification on synthetic batches")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs per ablation run (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 / 0.0001)")
    parser.add_argument("--device", type=str, default=None, help="Computing device ('cuda' or 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Skip already completed ablation cases")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case names to run exclusively")
    parser.add_argument("--no-hf-upload", action="store_true", help="Disable Hugging Face automatic upload")
    parser.add_argument("--hf-branch", type=str, default="main", help="HF target branch name")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.json")

    args = parser.parse_args()

    cases_filter = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None

    run_audio_ablation_suite(
        dry_run=args.dry_run,
        epochs=args.epochs,
        learning_rate=args.lr,
        device_str=args.device,
        resume=args.resume,
        cases_filter=cases_filter,
        hf_upload=not args.no_hf_upload,
        hf_branch=args.hf_branch,
        train_config_path=args.config,
    )


if __name__ == "__main__":
    main()
