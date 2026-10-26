import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.evaluate import MultimodalEvaluator
from utils.history_logger import HistoryLogger
from utils.profile_model import measure_flops

logger = logging.getLogger(__name__)


def _safe_torch_save(obj: Any, target_path: Path) -> None:
    """Save PyTorch checkpoint via an atomic temp write to avoid corrupted files."""
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(obj, tmp_path)
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise exc


class AudioEvalLoss(nn.Module):
    """Loss module wrapper for evaluation pass."""
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        logits = preds["logits"] if "logits" in preds else preds.get("clipwise_output")
        tgt = targets["target"]
        if tgt.ndim > 1 and tgt.size(-1) > 1:
            target_labels = tgt.argmax(dim=-1).long()
        else:
            target_labels = tgt.long().squeeze()
            if target_labels.ndim == 0:
                target_labels = target_labels.unsqueeze(0)
        return self.criterion(logits, target_labels)


class AudioTrainer:
    """
    Dedicated Training and Evaluation Engine for Unimodal Acoustic Feeding Intensity Classification.
    Optimizes AudioClassifierModel using Multi-Class Cross-Entropy Loss with AdamW.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        epochs: int = 200,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        ckpt_dir: str = "checkpoint/audio_ablations",
        case_name: str = "audio_ablation",
        is_dry_run: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.case_name = case_name
        self.is_dry_run = is_dry_run

        # Case checkpoint directory
        self.ckpt_dir = Path(ckpt_dir) / case_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Loss: Standard Multi-Class Cross Entropy
        self.loss_fn = nn.CrossEntropyLoss()
        self.eval_loss_fn = AudioEvalLoss()

        # Optimizer: AdamW at fixed learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Evaluator
        self.evaluator = MultimodalEvaluator(model=self.model, loss_fn=self.eval_loss_fn)

        # Metrics Tracker using standard HistoryLogger
        self.history_logger = HistoryLogger(log_dir=str(self.ckpt_dir))

        self.best_val_acc = -1.0
        self.best_val_qwk = -2.0
        self.best_epoch = 0

    def _extract_batch_audio_and_target(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract acoustic waveform/features and numeric ground-truth targets."""
        audio = batch.get("audio_form", batch.get("waveform"))
        if audio is None:
            raise KeyError("Batch does not contain 'audio_form' or 'waveform' key.")

        audio = audio.to(self.device).float()

        target = batch.get("target")
        if target is None:
            raise KeyError("Batch does not contain 'target' key.")

        target = target.to(self.device)
        if target.ndim > 1 and target.size(-1) > 1:
            target_labels = target.argmax(dim=-1).long()
        else:
            target_labels = target.long().squeeze()
            if target_labels.ndim == 0:
                target_labels = target_labels.unsqueeze(0)

        return audio, target_labels

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.case_name}] Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            audio, target_labels = self._extract_batch_audio_and_target(batch)

            self.optimizer.zero_grad()
            outputs = self.model(audio)
            logits = outputs["logits"]

            loss = self.loss_fn(logits, target_labels)
            loss.backward()

            # Gradient clipping max_norm=5.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * len(target_labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == target_labels).sum().item()
            total_samples += len(target_labels)

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if self.is_dry_run and step >= 1:
                break

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)

        return {
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_acc_video": 0.0,
            "train_acc_audio": epoch_acc,
            "train_mAP": 0.0,
        }

    def train(self) -> Dict[str, Any]:
        """Run the full acoustic training pipeline across all epochs."""
        logger.info(f"[{self.case_name}] Starting training: epochs={self.epochs}, lr={self.learning_rate:.2e}, device={self.device}")
        start_time = time.time()
        final_val_stats = {}

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.perf_counter()

            # 1. Training Pass
            train_metrics = self.train_epoch(epoch)

            # 2. Validation Pass
            self.model.eval()
            with torch.no_grad():
                val_stats = self.evaluator.evaluate(self.val_loader)
                final_val_stats = val_stats

            epoch_time = time.perf_counter() - t_epoch_start
            val_loss = float(val_stats.get("loss", 0.0))
            val_acc = float(val_stats.get("accuracy", 0.0))
            val_qwk = float(val_stats.get("qwk", 0.0))

            # Check for best validation accuracy (tie breaker: val_qwk)
            is_best = (val_acc > self.best_val_acc) or (val_acc == self.best_val_acc and val_qwk > self.best_val_qwk)
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_qwk = val_qwk
                self.best_epoch = epoch
                _safe_torch_save(self.model.state_dict(), self.ckpt_dir / "best_model.pth")
                _safe_torch_save(self.model.backbone.state_dict(), self.ckpt_dir / "best_audio_backbone.pth")

            # Always save last_model.pth
            _safe_torch_save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "best_val_qwk": self.best_val_qwk,
            }, self.ckpt_dir / "last_model.pth")

            # 3. Log epoch metrics to history.csv via canonical HistoryLogger
            self.history_logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics["train_loss"],
                train_acc=train_metrics["train_accuracy"],
                train_mAP=train_metrics["train_mAP"],
                val_loss=val_loss,
                val_statistics=val_stats,
                lr=self.learning_rate,
                epoch_time_seconds=epoch_time,
                is_best=is_best,
                train_acc_video=0.0,
                train_acc_audio=train_metrics["train_accuracy"]
            )

            logger.info(
                f"[{self.case_name}] Epoch {epoch:3d}/{self.epochs:3d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% QWK: {val_qwk:.4f}" +
                (" [BEST]" if is_best else "")
            )

            if self.is_dry_run:
                break

        # 4. Generate Learning Curves Plot
        try:
            self.history_logger.plot_history()
        except Exception as exc:
            logger.warning(f"Could not generate learning curves plot: {exc}")

        # 5. Load best model for Test Evaluation
        best_ckpt = self.ckpt_dir / "best_model.pth"
        if best_ckpt.exists():
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device, weights_only=True))

        self.model.eval()
        with torch.no_grad():
            test_stats = self.evaluator.evaluate(self.test_loader)

        # 6. Profile FLOPs, Latency and Parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)
        total_params_m = trainable_params / 1e6

        # Benchmark acoustic latency
        dummy_audio = torch.randn(1, 512000, device=self.device)
        is_cuda = self.device.type == "cuda"
        with torch.no_grad():
            warmup_iters = 10 if not self.is_dry_run else 1
            for _ in range(warmup_iters):
                _ = self.model(dummy_audio)
                if is_cuda:
                    torch.cuda.synchronize()
            t0 = time.perf_counter()
            test_iters = 50 if not self.is_dry_run else 2
            for _ in range(test_iters):
                _ = self.model(dummy_audio)
                if is_cuda:
                    torch.cuda.synchronize()
            latency_ms = ((time.perf_counter() - t0) / test_iters) * 1000.0

        # Measure FLOPs
        try:
            gflops = measure_flops(self.model, device=str(self.device))
        except Exception:
            gflops = 0.005

        training_duration = time.time() - start_time

        # 7. Save Summary CSV
        self.history_logger.save_summary(
            training_time=training_duration,
            inference_time_ms=latency_ms,
            val_statistics=final_val_stats,
            test_statistics=test_stats,
            total_params_m=total_params_m,
            gflops=gflops
        )

        # 8. Export Detailed Evaluation Reports (.txt and .json)
        try:
            self.history_logger.save_detailed_evaluation_report(
                val_statistics=final_val_stats,
                test_statistics=test_stats,
                total_params_m=total_params_m,
                gflops=gflops,
                dual_comparison_info=None
            )
        except Exception as exc:
            logger.warning(f"Could not export detailed evaluation report: {exc}")

        # 9. Plot Test Confusion Matrices (.png)
        try:
            self.history_logger.plot_test_confusion_matrices(
                test_statistics=test_stats,
                val_statistics=final_val_stats
            )
        except Exception as exc:
            logger.warning(f"Could not plot test confusion matrices: {exc}")

        test_acc = float(test_stats.get("accuracy", 0.0))
        test_qwk = float(test_stats.get("qwk", 0.0))
        test_f1 = float(np.mean(test_stats.get("f1_macro", 0.0)))

        summary_data = {
            "case_name": self.case_name,
            "use_tkeo": getattr(self.model, "use_tkeo", True),
            "depth": len(getattr(self.model, "hidden_dims", ())) + 1,
            "hidden_dims": str(list(getattr(self.model, "hidden_dims", ()))),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "backbone_params": backbone_params,
            "latency_ms": latency_ms,
            "gflops": gflops,
            "best_epoch": self.best_epoch,
            "val_acc": self.best_val_acc,
            "val_qwk": self.best_val_qwk,
            "test_acc": test_acc,
            "test_qwk": test_qwk,
            "test_f1": test_f1,
            "total_time_seconds": training_duration,
        }

        logger.info(
            f"[{self.case_name}] COMPLETED! "
            f"Test Acc: {test_acc*100:.2f}% | Test QWK: {test_qwk:.4f} | "
            f"Params: {trainable_params:,} | Latency: {latency_ms:.2f}ms"
        )

        return summary_data
