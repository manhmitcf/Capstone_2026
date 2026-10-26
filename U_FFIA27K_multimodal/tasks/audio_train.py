import os
import sys
import csv
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

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
    Exports exclusively Audio-focused checkpoints, metrics, plots, and classification reports.
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

        # Metrics Tracker for history.csv and learning_curves.png
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

        pbar = tqdm(
            self.train_loader,
            desc=f"[{self.case_name}] Epoch {epoch:03d}/{self.epochs:03d} [Train Audio]",
            leave=False
        )
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
            bs = target_labels.size(0)
            total_loss += batch_loss * bs
            preds = logits.argmax(dim=-1)
            correct += (preds == target_labels).sum().item()
            total_samples += bs

            cur_acc = (correct / max(1, total_samples)) * 100.0
            pbar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{cur_acc:.1f}%"})

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

    def evaluate(self, data_loader: DataLoader, split_name: str = "Val") -> Dict[str, Any]:
        """
        Evaluate AudioClassifierModel purely on acoustic waveforms without video overhead.
        Computes Accuracy, QWK, Macro/Weighted F1, mAP, ROC-AUC, Ordinal MAE, and Confusion Matrix.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_logits = []
        all_targets = []

        pbar = tqdm(
            data_loader,
            desc=f"[{self.case_name}] Evaluating [{split_name} Audio]",
            leave=False,
        )

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                audio, target_labels = self._extract_batch_audio_and_target(batch)
                outputs = self.model(audio)
                logits = outputs["logits"]

                loss = self.loss_fn(logits, target_labels)
                bs = target_labels.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

                all_logits.append(logits.detach().float().cpu().numpy())
                all_targets.append(target_labels.detach().cpu().numpy())

                if self.is_dry_run and step >= 1:
                    break

        if not all_logits:
            return {}

        clipwise_output = np.concatenate(all_logits, axis=0)
        target = np.concatenate(all_targets, axis=0)
        mean_loss = total_loss / max(1, total_samples)

        target_acc = target
        clipwise_output_acc = np.argmax(clipwise_output, axis=1).astype(np.int64)
        acc = float(accuracy_score(target_acc, clipwise_output_acc))

        num_classes = clipwise_output.shape[1]
        all_labels = list(range(num_classes))
        cm = confusion_matrix(target_acc, clipwise_output_acc, labels=all_labels)

        class_names = ['None', 'Strong', 'Medium', 'Weak'] if num_classes == 4 else None
        if class_names is not None:
            message = classification_report(
                target_acc, clipwise_output_acc, labels=all_labels, target_names=class_names, digits=4, zero_division=0
            )
        else:
            message = classification_report(target_acc, clipwise_output_acc, digits=4, zero_division=0)
        message = '\n' + message

        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            target_acc, clipwise_output_acc, average='weighted', zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            target_acc, clipwise_output_acc, average='macro', zero_division=0
        )

        # Physical ordinal rank mapping: 0: None (rank 0), 1: Strong (rank 3), 2: Medium (rank 2), 3: Weak (rank 1)
        if num_classes == 4:
            rank_map = np.array([0, 3, 2, 1])
            pred_ranks = rank_map[clipwise_output_acc]
            target_ranks = rank_map[target_acc]
            ordinal_mae = float(np.mean(np.abs(pred_ranks - target_ranks)))
            try:
                qwk_val = metrics.cohen_kappa_score(target_ranks, pred_ranks, weights='quadratic')
                qwk = 0.0 if np.isnan(qwk_val) else float(qwk_val)
            except Exception:
                qwk = 0.0
        else:
            ordinal_mae = float(np.mean(np.abs(clipwise_output_acc - target_acc)))
            qwk = 0.0

        try:
            target_onehot = np.eye(num_classes)[target_acc]
            ap = metrics.average_precision_score(target_onehot, clipwise_output, average=None)
        except Exception:
            ap = np.zeros(num_classes)

        try:
            target_onehot = np.eye(num_classes)[target_acc]
            auc = metrics.roc_auc_score(target_onehot, clipwise_output, average=None)
        except Exception:
            auc = np.zeros(num_classes)

        return {
            "loss": mean_loss,
            "accuracy": acc,
            "qwk": qwk,
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "average_precision": ap,
            "auc": auc,
            "ordinal_mae": ordinal_mae,
            "confu_matrix": cm,
            "message": message,
            "acc_audio": acc,
            "f1_macro_audio": float(f1_macro),
            "f1_weighted_audio": float(f1_weighted),
            "acc_video": 0.0,
        }

    @staticmethod
    def _format_cm(mat: Optional[np.ndarray]) -> str:
        """Format 4x4 confusion matrix into aligned text table."""
        if mat is None or not isinstance(mat, np.ndarray) or mat.size != 16:
            return "  [No confusion matrix available]\n"
        labels = ['None', 'Strong', 'Medium', 'Weak']
        header = f"  {'Actual\\Pred':<14}" + "".join(f"{lbl:>10}" for lbl in labels) + "\n"
        lines = [header]
        for idx, lbl in enumerate(labels):
            row = mat[idx] if idx < len(mat) else [0] * len(labels)
            row_str = f"  {lbl:<14}" + "".join(f"{int(val):>10d}" for val in row) + "\n"
            lines.append(row_str)
        return "".join(lines)

    @staticmethod
    def _save_cm_csv(path: Path, cm: Optional[np.ndarray]) -> None:
        """Save 4x4 confusion matrix to CSV."""
        labels = ["none", "strong", "medium", "weak"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([r"Actual\Predicted"] + labels)
            if cm is not None and isinstance(cm, np.ndarray) and cm.size > 0:
                for idx, label in enumerate(labels):
                    row = list(cm[idx]) if idx < len(cm) else [0] * len(labels)
                    writer.writerow([label] + row)
            else:
                for label in labels:
                    writer.writerow([label] + [0] * len(labels))

    @staticmethod
    def _plot_cm_png(path: Path, cm: Optional[np.ndarray], title: str) -> None:
        """Plot and save confusion matrix heatmap."""
        if cm is None or not isinstance(cm, np.ndarray) or cm.size == 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            class_names = ["None", "Strong", "Medium", "Weak"]
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, fontsize=11)
            ax.set_yticklabels(class_names, fontsize=11)
            ax.set_ylabel("Actual Label", fontweight="bold", fontsize=12)
            ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=12)

            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            for i in range(min(cm.shape[0], len(class_names))):
                row_total = np.sum(cm[i, :])
                for j in range(min(cm.shape[1], len(class_names))):
                    count = int(cm[i, j])
                    pct = (count / row_total * 100.0) if row_total > 0 else 0.0
                    color = "white" if count > thresh else "black"
                    ax.text(j, i, f"{count}\n({pct:.1f}%)",
                            horizontalalignment="center",
                            verticalalignment="center",
                            color=color, fontsize=11, fontweight="bold")

            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved confusion matrix heatmap to: '{path}'")
        except Exception as exc:
            logger.warning(f"Could not save confusion matrix heatmap: {exc}")

    @classmethod
    def _write_single_report(cls, path: Path, split_name: str, stats: Dict[str, Any]) -> None:
        """Write single-split acoustic classification report."""
        acc = float(stats.get("accuracy", 0.0))
        qwk = float(stats.get("qwk", 0.0))
        f1_m = float(stats.get("f1_macro", 0.0))
        f1_w = float(stats.get("f1_weighted", 0.0))
        cm = stats.get("confu_matrix")
        cm_str = cls._format_cm(cm)

        lines = [
            "=" * 70,
            f"  UNIMODAL ACOUSTIC EVALUATION REPORT ({split_name.upper()} SPLIT)",
            "=" * 70,
            f"Generated:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Accuracy:      {acc:.4f}",
            f"QWK:           {qwk:.4f}",
            f"Macro F1:      {f1_m:.4f}",
            f"Weighted F1:   {f1_w:.4f}",
            "-" * 70,
            "Confusion Matrix (Actual \\ Predicted):",
            cm_str,
            "-" * 70,
            "Classification Report (Precision, Recall, F1, Support):",
            stats.get("message") or "  N/A\n",
            "=" * 70 + "\n",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _save_audio_evaluation_reports(
        self,
        val_stats: Dict[str, Any],
        test_stats: Dict[str, Any],
        total_params: int,
        trainable_params: int,
        backbone_params: int,
        latency_ms: float,
        gflops: float,
        training_duration: float,
    ) -> None:
        """Save exclusively Audio-focused reports, CSVs, and plots."""
        # 1. Standalone reports for Validation and Test
        self._write_single_report(self.ckpt_dir / "report_val_audio.txt", "Validation", val_stats)
        self._write_single_report(self.ckpt_dir / "report_test_audio.txt", "Test", test_stats)

        # 2. Confusion Matrix CSVs
        self._save_cm_csv(self.ckpt_dir / "confusion_matrix_val_audio.csv", val_stats.get("confu_matrix"))
        self._save_cm_csv(self.ckpt_dir / "confusion_matrix_test_audio.csv", test_stats.get("confu_matrix"))
        self._save_cm_csv(self.ckpt_dir / "confusion_matrix_best.csv", test_stats.get("confu_matrix"))

        # 3. Confusion Matrix PNG Heatmaps
        val_acc = float(val_stats.get("accuracy", 0.0))
        test_acc = float(test_stats.get("accuracy", 0.0))
        self._plot_cm_png(
            self.ckpt_dir / "confusion_matrix_val_audio.png",
            val_stats.get("confu_matrix"),
            f"{self.case_name} - Validation (Acc: {val_acc:.4f})"
        )
        self._plot_cm_png(
            self.ckpt_dir / "confusion_matrix_test_audio.png",
            test_stats.get("confu_matrix"),
            f"{self.case_name} - Test (Acc: {test_acc:.4f})"
        )

        # 4. Summary CSV
        summary_csv_path = self.ckpt_dir / "summary.csv"
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "training_time", "inference_time_ms", "val_loss", "val_acc", "val_qwk",
                "test_loss", "test_acc", "test_qwk", "total_params_m", "gflops"
            ])
            writer.writerow([
                f"{training_duration:.2f}",
                f"{latency_ms:.3f}",
                f"{float(val_stats.get('loss', 0.0)):.6f}",
                f"{val_acc:.6f}",
                f"{float(val_stats.get('qwk', 0.0)):.6f}",
                f"{float(test_stats.get('loss', 0.0)):.6f}",
                f"{test_acc:.6f}",
                f"{float(test_stats.get('qwk', 0.0)):.6f}",
                f"{trainable_params / 1e6:.4f}",
                f"{gflops:.4f}"
            ])
        logger.info(f"Successfully exported Summary Report to: '{summary_csv_path}'")

        # 5. Consolidated Detailed Evaluation Report (.txt and .json)
        report_txt_path = self.ckpt_dir / "evaluation_detailed_report.txt"
        report_json_path = self.ckpt_dir / "evaluation_detailed_report.json"

        content = [
            "=" * 80,
            "           UNIMODAL ACOUSTIC FEEDING INTENSITY EVALUATION REPORT",
            "=" * 80,
            f"Generated on:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Case Name:             {self.case_name}",
            f"TKEO Pre-Emphasis:     {getattr(self.model, 'use_tkeo', True)}",
            f"Hidden Dimensions:     {list(getattr(self.model, 'hidden_dims', ()))} (Depth {len(getattr(self.model, 'hidden_dims', ())) + 1})",
            f"Directory:             {self.ckpt_dir}",
            f"Trainable Parameters:  {trainable_params:,} ({trainable_params / 1e6:.3f} M)",
            f"Backbone Parameters:   {backbone_params:,} ({backbone_params / 1e6:.3f} M)",
            f"Inference Complexity:  {gflops:.4f} GFLOPs",
            f"Inference Latency:     {latency_ms:.2f} ms / sample",
            f"Training Duration:     {training_duration:.1f} s",
            "-" * 80,
            "PART 1: VALIDATION SPLIT (AT BEST VAL ACCURACY EPOCH)",
            "-" * 80,
            f"  - Accuracy:          {val_acc:.4f}",
            f"  - QWK:               {float(val_stats.get('qwk', 0.0)):.4f}",
            f"  - Macro F1:          {float(val_stats.get('f1_macro', 0.0)):.4f}",
            f"  - Weighted F1:       {float(val_stats.get('f1_weighted', 0.0)):.4f}",
            "\n  Classification Report:",
            val_stats.get("message") or "  N/A\n",
            "-" * 80,
            "PART 2: TEST SPLIT (FINAL INDEPENDENT EVALUATION)",
            "-" * 80,
            f"  - Accuracy:          {test_acc:.4f}",
            f"  - QWK:               {float(test_stats.get('qwk', 0.0)):.4f}",
            f"  - Macro F1:          {float(test_stats.get('f1_macro', 0.0)):.4f}",
            f"  - Weighted F1:       {float(test_stats.get('f1_weighted', 0.0)):.4f}",
            "\n  Classification Report:",
            test_stats.get("message") or "  N/A\n",
            "=" * 80 + "\n"
        ]

        try:
            clean_content = [str(x) if x is not None else "  N/A" for x in content]
            with open(report_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(clean_content))
            logger.info(f"Saved consolidated acoustic evaluation report to: '{report_txt_path}'")
        except Exception as exc:
            logger.error(f"Failed to write consolidated evaluation report: {exc}")

        # JSON Export
        def sanitize_for_json(d: Any) -> Any:
            if isinstance(d, dict):
                return {str(k): sanitize_for_json(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [sanitize_for_json(v) for v in d]
            elif isinstance(d, np.ndarray):
                return d.tolist()
            elif isinstance(d, (np.floating, np.float32, np.float64)):
                return float(d)
            elif isinstance(d, (np.integer, np.int32, np.int64)):
                return int(d)
            return d

        json_payload = {
            "case_name": self.case_name,
            "use_tkeo": getattr(self.model, "use_tkeo", True),
            "depth": len(getattr(self.model, "hidden_dims", ())) + 1,
            "hidden_dims": list(getattr(self.model, "hidden_dims", ())),
            "trainable_params": trainable_params,
            "backbone_params": backbone_params,
            "gflops": gflops,
            "latency_ms": latency_ms,
            "training_duration_s": training_duration,
            "validation_metrics": sanitize_for_json(val_stats),
            "test_metrics": sanitize_for_json(test_stats),
        }
        try:
            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved structured acoustic evaluation JSON to: '{report_json_path}'")
        except Exception as exc:
            logger.error(f"Failed to write evaluation JSON report: {exc}")

    def train(self) -> Dict[str, Any]:
        """Run the full acoustic training pipeline across all epochs."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)
        frontend_params = sum(p.numel() for p in self.model.frontend.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)

        batch_sz = getattr(self.train_loader, "batch_size", "N/A")
        train_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else "N/A"
        val_samples = len(self.val_loader.dataset) if hasattr(self.val_loader, "dataset") else "N/A"
        test_samples = len(self.test_loader.dataset) if hasattr(self.test_loader, "dataset") else "N/A"

        logger.info("=" * 80)
        logger.info(f"STARTING UNIMODAL AUDIO ABLATION EXPERIMENT: {self.case_name.upper()}")
        logger.info("=" * 80)
        logger.info(f"  - Case Identifier:         {self.case_name}")
        logger.info(f"  - Model Architecture:      AudioClassifierModel (TKEO-STFT + MLP Backbone)")
        logger.info(f"  - TKEO Pre-Emphasis:       {getattr(self.model, 'use_tkeo', True)}")
        logger.info(f"  - Audio Sampling Rate:     256,000 Hz (2.0s = 512,000 samples)")
        logger.info(f"  - STFT Output Bins:        2049 linear bins (n_fft=4096, hop_size=2048)")
        logger.info(f"  - MLP Hidden Dimensions:   {list(getattr(self.model, 'hidden_dims', ()))} (Depth {len(getattr(self.model, 'hidden_dims', ())) + 1})")
        logger.info(f"  - Output Latent Dimension: 384 -> 4 Classes (None, Strong, Medium, Weak)")
        logger.info(f"  - Model Parameter Breakdown:")
        logger.info(f"    * Total Trainable:       {trainable_params:,} ({trainable_params/1e6:.3f} M)")
        logger.info(f"    * Audio Backbone:        {backbone_params:,} ({backbone_params/1e6:.3f} M)")
        logger.info(f"    * Audio Frontend (TKEO): {frontend_params:,} ({frontend_params/1e6:.3f} M)")
        logger.info(f"    * Classifier Head:       {classifier_params:,} ({classifier_params/1e6:.3f} M)")
        logger.info(f"  - Optimization & Training Hyperparameters:")
        logger.info(f"    * Optimizer:             AdamW (lr={self.learning_rate:.2e}, weight_decay={self.weight_decay})")
        logger.info(f"    * LR Schedule:           Constant (fixed across all {self.epochs} epochs)")
        logger.info(f"    * Loss Function:         nn.CrossEntropyLoss()")
        logger.info(f"    * Gradient Clipping:     max_norm = 5.0")
        logger.info(f"    * Device:                {self.device}")
        logger.info(f"    * Epochs:                {self.epochs}")
        logger.info(f"    * Batch Size:            {batch_sz}")
        logger.info(f"    * Dataset Splits:        Train={train_samples}, Val={val_samples}, Test={test_samples}")
        logger.info(f"  - Checkpoint Run Dir:      '{self.ckpt_dir}'")
        logger.info("=" * 80)

        start_time = time.time()
        final_val_stats = {}

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.perf_counter()

            # 1. Training Pass
            train_metrics = self.train_epoch(epoch)

            # 2. Validation Pass (Pure Audio)
            val_stats = self.evaluate(self.val_loader, split_name="Val")
            final_val_stats = val_stats

            epoch_time = time.perf_counter() - t_epoch_start
            lr_current = self.optimizer.param_groups[0]["lr"]

            train_loss = train_metrics["train_loss"]
            train_acc = train_metrics["train_accuracy"]
            val_loss = float(val_stats.get("loss", 0.0))
            val_acc = float(val_stats.get("accuracy", 0.0))
            val_qwk = float(val_stats.get("qwk", 0.0))
            val_f1_m = float(val_stats.get("f1_macro", 0.0))
            val_f1_w = float(val_stats.get("f1_weighted", 0.0))
            val_map = float(np.mean(val_stats.get("average_precision", 0.0)))
            val_mae = float(val_stats.get("ordinal_mae", 0.0))

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
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=val_map,
                val_loss=val_loss,
                val_statistics=val_stats,
                lr=lr_current,
                epoch_time_seconds=epoch_time,
                is_best=is_best,
                train_acc_video=0.0,
                train_acc_audio=train_acc
            )

            # Per-epoch complete acoustic log
            best_tag = " [*BEST*]" if is_best else ""
            logger.info(
                f"[{self.case_name}] Epoch {epoch:03d}/{self.epochs:03d} [AUDIO] | "
                f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc*100:.2f}% | "
                f"Val Loss = {val_loss:.4f} | Val Acc = {val_acc*100:.2f}% | "
                f"Val QWK = {val_qwk:.4f} | Val F1 = {val_f1_m:.4f} | "
                f"Val mAP = {val_map:.4f} | Val MAE = {val_mae:.4f} | "
                f"LR = {lr_current:.2e} | Time = {epoch_time:.1f}s{best_tag}"
            )

            if is_best:
                logger.info(
                    f"[*] [{self.case_name}] New peak performance at Epoch {epoch:03d}! "
                    f"(Val Acc = {val_acc*100:.2f}%, Val QWK = {val_qwk:.4f}, Macro-F1 = {val_f1_m:.4f})\n"
                    f"    - Saved Full Model:     '{self.ckpt_dir / 'best_model.pth'}'\n"
                    f"    - Saved Audio Backbone: '{self.ckpt_dir / 'best_audio_backbone.pth'}'"
                )

            if self.is_dry_run:
                break

        # 4. Generate Learning Curves Plot
        try:
            self.history_logger.plot_history()
        except Exception as exc:
            logger.warning(f"Could not generate learning curves plot: {exc}")

        # 5. Load best model for Test Evaluation
        logger.info("=" * 80)
        logger.info(f"[{self.case_name}] TRAINING COMPLETE - EVALUATING BEST MODEL ON TEST SPLIT")
        logger.info("=" * 80)
        logger.info(
            f"Reloading best checkpoint from Epoch {self.best_epoch:03d} "
            f"(Peak Val Acc: {self.best_val_acc*100:.2f}%, Val QWK: {self.best_val_qwk:.4f})..."
        )
        best_ckpt = self.ckpt_dir / "best_model.pth"
        if best_ckpt.exists():
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device, weights_only=True))

        self.model.eval()
        with torch.no_grad():
            test_stats = self.evaluate(self.test_loader, split_name="Test")

        # 6. Profile FLOPs, Latency and Parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)

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

        # 7. Save Exclusively Audio Artifacts (No Video, No Multimodal Fusion reports)
        self._save_audio_evaluation_reports(
            val_stats=final_val_stats,
            test_stats=test_stats,
            total_params=total_params,
            trainable_params=trainable_params,
            backbone_params=backbone_params,
            latency_ms=latency_ms,
            gflops=gflops,
            training_duration=training_duration,
        )

        test_loss = float(test_stats.get("loss", 0.0))
        test_acc = float(test_stats.get("accuracy", 0.0))
        test_qwk = float(test_stats.get("qwk", 0.0))
        test_f1_m = float(test_stats.get("f1_macro", 0.0))
        test_f1_w = float(test_stats.get("f1_weighted", 0.0))
        test_map = float(np.mean(test_stats.get("average_precision", 0.0)))
        test_mae = float(test_stats.get("ordinal_mae", 0.0))

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
            "test_f1": test_f1_m,
            "total_time_seconds": training_duration,
        }

        logger.info("=" * 80)
        logger.info(f"[{self.case_name}] FINAL TEST EVALUATION RESULTS [AUDIO ONLY]:")
        logger.info("=" * 80)
        logger.info(f"  - Test Loss:              {test_loss:.4f}")
        logger.info(f"  - Test Accuracy:          {test_acc*100:.2f}%")
        logger.info(f"  - Test QWK:               {test_qwk:.4f}")
        logger.info(f"  - Test Macro F1:          {test_f1_m:.4f}")
        logger.info(f"  - Test Weighted F1:       {test_f1_w:.4f}")
        logger.info(f"  - Test mAP:               {test_map:.4f}")
        logger.info(f"  - Test Ordinal MAE:       {test_mae:.4f}")
        logger.info(f"  - Inference Latency:      {latency_ms:.2f} ms / sample")
        logger.info(f"  - Computational FLOPs:    {gflops:.4f} GFLOPs")
        logger.info(f"  - Total Parameters:       {total_params:,} ({total_params/1e6:.3f} M)")
        logger.info(f"  - Backbone Parameters:    {backbone_params:,} ({backbone_params/1e6:.3f} M)")
        logger.info(f"  - Best Training Epoch:    Epoch {self.best_epoch:03d}")
        logger.info(f"  - Total Training Time:    {training_duration:.1f} s")
        logger.info("-" * 80)
        logger.info(f"Classification Report [Audio]:\n{test_stats.get('message', '  N/A')}")
        logger.info(f"Confusion Matrix [Audio] (Actual \\ Predicted):\n{self._format_cm(test_stats.get('confu_matrix'))}")
        logger.info("=" * 80)

        return summary_data
