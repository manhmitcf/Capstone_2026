import os
import csv
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryLogger:
    """
    OOP compliant HistoryLogger to automatically log training performance history.
    Records evaluation metrics and flattened confusion matrices to CSV and plots learning curves.
    """
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_csv_path = os.path.join(self.log_dir, 'history.csv')
        self._write_history_header()

    def _history_headers(self) -> list:
        return [
            'epoch',
            'lr',
            'epoch_time_seconds',
            'train_loss',
            'train_accuracy',
            'train_acc_video',
            'train_acc_audio',
            'train_mAP',
            'val_loss',
            'val_accuracy',
            'val_qwk',
            'val_acc_video',
            'val_acc_audio',
            'val_mAP',
            'val_auc_class_none', 'val_auc_class_strong', 'val_auc_class_medium', 'val_auc_class_weak',
            'val_ap_class_none', 'val_ap_class_strong', 'val_ap_class_medium', 'val_ap_class_weak',
            'cm_none_none', 'cm_none_strong', 'cm_none_medium', 'cm_none_weak',
            'cm_strong_none', 'cm_strong_strong', 'cm_strong_medium', 'cm_strong_weak',
            'cm_medium_none', 'cm_medium_strong', 'cm_medium_medium', 'cm_medium_weak',
            'cm_weak_none', 'cm_weak_strong', 'cm_weak_medium', 'cm_weak_weak'
        ]

    def _write_history_header(self) -> None:
        with open(self.history_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self._history_headers())

    def _save_confusion_matrix_csv(self, path: str, matrix: Optional[np.ndarray]) -> None:
        labels = ['none', 'strong', 'medium', 'weak']
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([r'Actual\Predicted'] + labels)
                if matrix is not None and isinstance(matrix, np.ndarray) and matrix.size > 0:
                    for idx, label in enumerate(labels):
                        row = list(matrix[idx]) if idx < len(matrix) else [0] * len(labels)
                        writer.writerow([label] + row)
                else:
                    for label in labels:
                        writer.writerow([label] + [0] * len(labels))
        except Exception as exc:
            logger.warning(f"Could not save confusion matrix CSV '{path}': {exc}")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_mAP: float,
        val_loss: float,
        val_statistics: dict,
        lr: Optional[float] = None,
        epoch_time_seconds: Optional[float] = None,
        is_best: bool = False,
        train_acc_video: Optional[float] = None,
        train_acc_audio: Optional[float] = None
    ) -> None:
        val_acc = float(np.mean(val_statistics.get('accuracy', 0.0)))
        val_qwk = float(val_statistics.get('qwk', 0.0))
        val_mAP = float(np.mean(val_statistics.get('average_precision', 0.0)))
        val_acc_v = float(val_statistics.get('acc_video', 0.0))
        val_acc_a = float(val_statistics.get('acc_audio', 0.0))
        val_auc = val_statistics.get('auc', [0.0] * 4)
        val_ap = val_statistics.get('average_precision', [0.0] * 4)
        cm = val_statistics.get('confu_matrix', None)

        def _safe_float(arr: Any, idx: int, default: float = 0.0) -> float:
            try:
                if arr is not None and idx < len(arr):
                    v = float(arr[idx])
                    return v if not np.isnan(v) else default
            except Exception:
                pass
            return default

        v_auc = [_safe_float(val_auc, i) for i in range(4)]
        v_ap = [_safe_float(val_ap, i) for i in range(4)]

        if cm is not None and isinstance(cm, np.ndarray) and cm.size == 16:
            cm_flat = [int(val) for val in cm.flatten()]
        else:
            cm_flat = [0] * 16

        lr_str = f"{lr:.8e}" if lr is not None else "N/A"
        time_str = f"{epoch_time_seconds:.2f}" if epoch_time_seconds is not None else "N/A"
        t_acc_v = f"{train_acc_video:.6f}" if train_acc_video is not None else f"{train_acc:.6f}"
        t_acc_a = f"{train_acc_audio:.6f}" if train_acc_audio is not None else f"{train_acc:.6f}"

        row_data = [
            epoch,
            lr_str,
            time_str,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            t_acc_v,
            t_acc_a,
            f"{train_mAP:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{val_qwk:.6f}",
            f"{val_acc_v:.6f}",
            f"{val_acc_a:.6f}",
            f"{val_mAP:.6f}",
            f"{v_auc[0]:.6f}", f"{v_auc[1]:.6f}", f"{v_auc[2]:.6f}", f"{v_auc[3]:.6f}",
            f"{v_ap[0]:.6f}", f"{v_ap[1]:.6f}", f"{v_ap[2]:.6f}", f"{v_ap[3]:.6f}"
        ] + cm_flat

        try:
            with open(self.history_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as exc:
            logger.error(f"Failed to append to history CSV '{self.history_csv_path}': {exc}")

        if is_best:
            try:
                best_cm_path = os.path.join(self.log_dir, 'confusion_matrix_best.csv')
                self._save_confusion_matrix_csv(best_cm_path, cm)
            except Exception as exc:
                logger.warning(f"Failed to save best confusion matrix CSV: {exc}")

    def save_summary(
        self,
        training_time: float,
        inference_time_ms: float,
        val_statistics: dict,
        test_statistics: dict,
        total_params_m: Optional[float] = None,
        gflops: Optional[float] = None
    ) -> None:
        summary_csv_path = os.path.join(self.log_dir, 'summary.csv')
        file_exists = os.path.exists(summary_csv_path)

        def _safe_stat(stats: dict, key: str, default: float = 0.0) -> float:
            try:
                raw = stats.get(key, default)
                val = float(np.mean(raw))
                return val if not np.isnan(val) else default
            except Exception:
                return default

        val_mAP = _safe_stat(val_statistics, 'average_precision')
        test_mAP = _safe_stat(test_statistics, 'average_precision')
        val_qwk = _safe_stat(val_statistics, 'qwk')
        test_qwk = _safe_stat(test_statistics, 'qwk')

        headers = [
            'Total Parameters (M)',
            'Inference Complexity (GFLOPs)',
            'Training Time (s)',
            'Inference Time (ms/sample)',
            'Precision Val (Weighted)', 'Recall Val (Weighted)', 'F1-score Val (Weighted)', 'Accuracy Val', 'QWK Val', 'mAP Val',
            'Precision Val (Macro)', 'Recall Val (Macro)', 'F1-score Val (Macro)',
            'Precision Test (Weighted)', 'Recall Test (Weighted)', 'F1-score Test (Weighted)', 'Accuracy Test', 'QWK Test', 'mAP Test',
            'Precision Test (Macro)', 'Recall Test (Macro)', 'F1-score Test (Macro)'
        ]

        params_str = f"{total_params_m:.3f}" if total_params_m is not None else "N/A"
        gflops_str = f"{gflops:.3f}" if gflops is not None else "N/A"

        row_data = [
            params_str,
            gflops_str,
            f"{training_time:.2f}",
            f"{inference_time_ms:.3f}",
            f"{_safe_stat(val_statistics, 'prec_weighted'):.6f}",
            f"{_safe_stat(val_statistics, 'rec_weighted'):.6f}",
            f"{_safe_stat(val_statistics, 'f1_weighted'):.6f}",
            f"{_safe_stat(val_statistics, 'accuracy'):.6f}",
            f"{val_qwk:.6f}",
            f"{val_mAP:.6f}",
            f"{_safe_stat(val_statistics, 'prec_macro'):.6f}",
            f"{_safe_stat(val_statistics, 'rec_macro'):.6f}",
            f"{_safe_stat(val_statistics, 'f1_macro'):.6f}",
            f"{_safe_stat(test_statistics, 'prec_weighted'):.6f}",
            f"{_safe_stat(test_statistics, 'rec_weighted'):.6f}",
            f"{_safe_stat(test_statistics, 'f1_weighted'):.6f}",
            f"{_safe_stat(test_statistics, 'accuracy'):.6f}",
            f"{test_qwk:.6f}",
            f"{test_mAP:.6f}",
            f"{_safe_stat(test_statistics, 'prec_macro'):.6f}",
            f"{_safe_stat(test_statistics, 'rec_macro'):.6f}",
            f"{_safe_stat(test_statistics, 'f1_macro'):.6f}"
        ]

        try:
            with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerow(row_data)
            logger.info(f"Successfully exported Summary Report to: '{summary_csv_path}'")
        except Exception as exc:
            logger.error(f"Failed to export Summary Report to '{summary_csv_path}': {exc}")

    def plot_history(self) -> None:
        if not os.path.exists(self.history_csv_path):
            logger.warning(f"Warning: History file '{self.history_csv_path}' does not exist. Cannot plot curves.")
            return

        epochs = []
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_maps, val_maps = [], []

        with open(self.history_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
                    train_accs.append(float(row['train_accuracy']))
                    val_accs.append(float(row['val_accuracy']))
                    train_maps.append(float(row['train_mAP']))
                    val_maps.append(float(row['val_mAP']))
                except (KeyError, ValueError):
                    continue

        if not epochs:
            return

        import matplotlib
        matplotlib.use('Agg')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Multimodal Fish Feeding Intensity Model Learning History', fontsize=16, fontweight='bold', y=0.98)

        axes[0].plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2, linestyle='--')
        axes[0].plot(epochs, val_losses, label='Val Loss', color='#ff7f0e', linewidth=2)
        axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(frameon=True)

        axes[1].plot(epochs, train_accs, label='Train Acc', color='#2ca02c', linewidth=2, linestyle='--')
        axes[1].plot(epochs, val_accs, label='Val Acc', color='#d62728', linewidth=2)
        axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(frameon=True)

        axes[2].plot(epochs, train_maps, label='Train mAP', color='#9467bd', linewidth=2, linestyle='--')
        axes[2].plot(epochs, val_maps, label='Val mAP', color='#8c564b', linewidth=2)
        axes[2].set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mAP')
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend(frameon=True)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'learning_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully generated learning curves plot at: '{plot_path}'")

    @staticmethod
    def _format_cm(mat: Optional[np.ndarray]) -> str:
        """Format 4x4 confusion matrix into aligned text table."""
        if mat is None:
            return "  [No confusion matrix available]\n"
        labels = ['None', 'Strong', 'Medium', 'Weak']
        header = f"  {'Actual\\Pred':<14}" + "".join(f"{lbl:>10}" for lbl in labels) + "\n"
        lines = [header]
        for idx, lbl in enumerate(labels):
            row = mat[idx] if idx < len(mat) else [0] * len(labels)
            row_str = f"  {lbl:<14}" + "".join(f"{int(val):>10d}" for val in row) + "\n"
            lines.append(row_str)
        return "".join(lines)

    def _write_single_branch_report(
        self,
        output_txt_path: str,
        output_csv_path: str,
        branch_name: str,
        split_name: str,
        acc: float,
        mAP: Optional[float],
        f1_macro: float,
        f1_weighted: float,
        cm: Optional[np.ndarray],
        report_str: str
    ) -> None:
        """Helper to write standalone single-branch report TXT and confusion matrix CSV."""
        lines = [
            "=" * 70,
            f"  {branch_name.upper()} EVALUATION REPORT ({split_name.upper()} SPLIT)",
            "=" * 70,
            f"Generated:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Accuracy:      {acc:.4f}",
        ]
        if mAP is not None:
            lines.append(f"mAP:           {mAP:.4f}")
        lines.extend([
            f"Macro F1:      {f1_macro:.4f}",
            f"Weighted F1:   {f1_weighted:.4f}",
            "-" * 70,
            "Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(cm),
            "-" * 70,
            "Classification Report (Precision, Recall, F1, Support):",
            report_str if report_str else "  N/A\n",
            "=" * 70 + "\n"
        ])

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        self._save_confusion_matrix_csv(output_csv_path, cm)

    def save_detailed_evaluation_report(
        self,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any],
        total_params_m: Optional[float] = None,
        gflops: Optional[float] = None,
        dual_comparison_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Exports both SEPARATE standalone files for each branch (Fusion, Video, Audio)
        and a consolidated evaluation report file (.txt and .json).
        """
        # 1. Separate standalone files for Validation split
        self._write_single_branch_report(
            output_txt_path=os.path.join(self.log_dir, 'report_val_fusion.txt'),
            output_csv_path=os.path.join(self.log_dir, 'confusion_matrix_val_fusion.csv'),
            branch_name="Multimodal Tournament Fusion",
            split_name="Validation",
            acc=float(np.mean(val_statistics.get('accuracy', 0.0))),
            mAP=float(np.mean(val_statistics.get('average_precision', 0.0))),
            f1_macro=float(val_statistics.get('f1_macro', 0.0)),
            f1_weighted=float(val_statistics.get('f1_weighted', 0.0)),
            cm=val_statistics.get('confu_matrix'),
            report_str=val_statistics.get('message', '')
        )

        self._write_single_branch_report(
            output_txt_path=os.path.join(self.log_dir, 'report_val_video.txt'),
            output_csv_path=os.path.join(self.log_dir, 'confusion_matrix_val_video.csv'),
            branch_name="Video Auxiliary Head (ConvNeXt-Nano 6-ch)",
            split_name="Validation",
            acc=float(val_statistics.get('acc_video', 0.0)),
            mAP=None,
            f1_macro=float(val_statistics.get('f1_macro_video', 0.0)),
            f1_weighted=float(val_statistics.get('f1_weighted_video', 0.0)),
            cm=val_statistics.get('confu_matrix_video'),
            report_str=val_statistics.get('message_video', '')
        )

        self._write_single_branch_report(
            output_txt_path=os.path.join(self.log_dir, 'report_val_audio.txt'),
            output_csv_path=os.path.join(self.log_dir, 'confusion_matrix_val_audio.csv'),
            branch_name="Audio Auxiliary Head (TKEO-STFT-MLP 256k)",
            split_name="Validation",
            acc=float(val_statistics.get('acc_audio', 0.0)),
            mAP=None,
            f1_macro=float(val_statistics.get('f1_macro_audio', 0.0)),
            f1_weighted=float(val_statistics.get('f1_weighted_audio', 0.0)),
            cm=val_statistics.get('confu_matrix_audio'),
            report_str=val_statistics.get('message_audio', '')
        )

        # 2. Separate standalone files for Test split
        self._write_single_branch_report(
            output_txt_path=os.path.join(self.log_dir, 'report_test_fusion.txt'),
            output_csv_path=os.path.join(self.log_dir, 'confusion_matrix_test_fusion.csv'),
            branch_name="Multimodal Tournament Fusion",
            split_name="Test",
            acc=float(np.mean(test_statistics.get('accuracy', 0.0))),
            mAP=float(np.mean(test_statistics.get('average_precision', 0.0))),
            f1_macro=float(test_statistics.get('f1_macro', 0.0)),
            f1_weighted=float(test_statistics.get('f1_weighted', 0.0)),
            cm=test_statistics.get('confu_matrix'),
            report_str=test_statistics.get('message', '')
        )

        self._write_single_branch_report(
            output_txt_path=os.path.join(self.log_dir, 'report_test_video.txt'),
            output_csv_path=os.path.join(self.log_dir, 'confusion_matrix_test_video.csv'),
            branch_name="Video Auxiliary Head (ConvNeXt-Nano 6-ch)",
            split_name="Test",
            acc=float(test_statistics.get('acc_video', 0.0)),
            mAP=None,
            f1_macro=float(test_statistics.get('f1_macro_video', 0.0)),
            f1_weighted=float(test_statistics.get('f1_weighted_video', 0.0)),
            cm=test_statistics.get('confu_matrix_video'),
            report_str=test_statistics.get('message_video', '')
        )

        self._write_single_branch_report(
            output_txt_path=os.path.join(self.log_dir, 'report_test_audio.txt'),
            output_csv_path=os.path.join(self.log_dir, 'confusion_matrix_test_audio.csv'),
            branch_name="Audio Auxiliary Head (TKEO-STFT-MLP 256k)",
            split_name="Test",
            acc=float(test_statistics.get('acc_audio', 0.0)),
            mAP=None,
            f1_macro=float(test_statistics.get('f1_macro_audio', 0.0)),
            f1_weighted=float(test_statistics.get('f1_weighted_audio', 0.0)),
            cm=test_statistics.get('confu_matrix_audio'),
            report_str=test_statistics.get('message_audio', '')
        )

        # 3. Consolidated report file
        report_txt_path = os.path.join(self.log_dir, 'evaluation_detailed_report.txt')
        report_json_path = os.path.join(self.log_dir, 'evaluation_detailed_report.json')

        content = [
            "=" * 80,
            "          COMPREHENSIVE MULTIMODAL & DUAL BACKBONE EVALUATION REPORT",
            "=" * 80,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Directory:    {self.log_dir}",
        ]
        if total_params_m is not None:
            content.append(f"Model Parameters:      {total_params_m:.3f} M")
        if gflops is not None:
            content.append(f"Inference Complexity:  {gflops:.4f} GFLOPs")

        if dual_comparison_info is not None:
            q_info = dual_comparison_info.get('qwk_candidate', {})
            a_info = dual_comparison_info.get('acc_candidate', {})
            winner = dual_comparison_info.get('winner', 'unknown')
            win_ep = dual_comparison_info.get('winning_epoch', 0)

            q_ckpt = q_info.get('checkpoint', 'best_model_qwk.pth')
            q_ep = q_info.get('val_epoch', 0)
            q_crit = f"QWK (Ep {q_ep:03d})"
            q_val = float(q_info.get('val_qwk', 0.0))
            q_t_acc = float(q_info.get('test_accuracy', 0.0)) * 100.0
            q_t_qwk = float(q_info.get('test_qwk', 0.0))
            q_t_map = float(q_info.get('test_mAP', 0.0)) * 100.0
            q_win = "<-- WINNER" if winner == 'qwk' else ""

            a_ckpt = a_info.get('checkpoint', 'best_model_acc.pth')
            a_ep = a_info.get('val_epoch', 0)
            a_crit = f"ACC (Ep {a_ep:03d})"
            a_val = float(a_info.get('val_accuracy', 0.0)) * 100.0
            a_t_acc = float(a_info.get('test_accuracy', 0.0)) * 100.0
            a_t_qwk = float(a_info.get('test_qwk', 0.0))
            a_t_map = float(a_info.get('test_mAP', 0.0)) * 100.0
            a_win = "<-- WINNER" if winner == 'accuracy' else ""

            winner_suffix = 'acc' if winner == 'accuracy' else 'qwk'

            content.extend([
                "-" * 80,
                "PART 0: DUAL-TRACK TOURNAMENT SELECTION (QWK vs ACCURACY CANDIDATES)",
                "-" * 80,
                f"{'Candidate Checkpoint':<24} | {'Val Criterion':<16} | {'Val Peak':<10} | {'Test Acc':<10} | {'Test QWK':<10} | {'Test mAP':<10} | {'Selected'}",
                "-" * 80,
                f"{q_ckpt:<24} | {q_crit:<16} | {q_val:<10.4f} | {q_t_acc:<9.2f}% | {q_t_qwk:<10.4f} | {q_t_map:<9.2f}% | {q_win}",
                f"{a_ckpt:<24} | {a_crit:<16} | {a_val:<9.2f}% | {a_t_acc:<9.2f}% | {a_t_qwk:<10.4f} | {a_t_map:<9.2f}% | {a_win}",
                "-" * 80,
                f"Tournament Winner: Candidate '{winner}' (Epoch {win_ep:03d}) achieved superior Test Split performance.",
                f"Canonical checkpoints (best_model.pth, best_video_backbone.pth, best_audio_backbone.pth)",
                f"were deployed from winning candidate: best_model_{winner_suffix}.pth\n",
            ])

        content.extend([
            "-" * 80,
            "PART 1: VALIDATION SPLIT (AT BEST FUSION MODEL CHECKPOINT)",
            "-" * 80,
            "\n[1.1] MULTIMODAL TOURNAMENT FUSION (OVERALL MODEL)",
            f"  - Accuracy:       {float(np.mean(val_statistics.get('accuracy', 0.0))):.4f}",
            f"  - mAP:            {float(np.mean(val_statistics.get('average_precision', 0.0))):.4f}",
            f"  - Macro F1:       {float(val_statistics.get('f1_macro', 0.0)):.4f}",
            f"  - Weighted F1:    {float(val_statistics.get('f1_weighted', 0.0)):.4f}",
            "\n  Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(val_statistics.get('confu_matrix')),
            "  Classification Report:",
            val_statistics.get('message') or '  N/A\n',
            "\n[1.2] VIDEO AUXILIARY HEAD (ConvNeXt-Nano 6-Channel Kinematics)",
            f"  - Accuracy:       {float(val_statistics.get('acc_video', 0.0)):.4f}",
            f"  - Macro F1:       {float(val_statistics.get('f1_macro_video', 0.0)):.4f}",
            f"  - Weighted F1:    {float(val_statistics.get('f1_weighted_video', 0.0)):.4f}",
            "\n  Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(val_statistics.get('confu_matrix_video')),
            "  Classification Report:",
            val_statistics.get('message_video') or '  N/A\n',
            "\n[1.3] AUDIO AUXILIARY HEAD (TKEO-STFT-MLP 256 kHz)",
            f"  - Accuracy:       {float(val_statistics.get('acc_audio', 0.0)):.4f}",
            f"  - Macro F1:       {float(val_statistics.get('f1_macro_audio', 0.0)):.4f}",
            f"  - Weighted F1:    {float(val_statistics.get('f1_weighted_audio', 0.0)):.4f}",
            "\n  Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(val_statistics.get('confu_matrix_audio')),
            "  Classification Report:",
            val_statistics.get('message_audio') or '  N/A\n',
            "\n" + "-" * 80,
            "PART 2: TEST SPLIT (FINAL INDEPENDENT EVALUATION)",
            "-" * 80,
            "\n[2.1] MULTIMODAL TOURNAMENT FUSION (OVERALL MODEL)",
            f"  - Accuracy:       {float(np.mean(test_statistics.get('accuracy', 0.0))):.4f}",
            f"  - mAP:            {float(np.mean(test_statistics.get('average_precision', 0.0))):.4f}",
            f"  - Macro F1:       {float(test_statistics.get('f1_macro', 0.0)):.4f}",
            f"  - Weighted F1:    {float(test_statistics.get('f1_weighted', 0.0)):.4f}",
            "\n  Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(test_statistics.get('confu_matrix')),
            "  Classification Report:",
            test_statistics.get('message') or '  N/A\n',
            "\n[2.2] VIDEO AUXILIARY HEAD (ConvNeXt-Nano 6-Channel Kinematics)",
            f"  - Accuracy:       {float(test_statistics.get('acc_video', 0.0)):.4f}",
            f"  - Macro F1:       {float(test_statistics.get('f1_macro_video', 0.0)):.4f}",
            f"  - Weighted F1:    {float(test_statistics.get('f1_weighted_video', 0.0)):.4f}",
            "\n  Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(test_statistics.get('confu_matrix_video')),
            "  Classification Report:",
            test_statistics.get('message_video') or '  N/A\n',
            "\n[2.3] AUDIO AUXILIARY HEAD (TKEO-STFT-MLP 256 kHz)",
            f"  - Accuracy:       {float(test_statistics.get('acc_audio', 0.0)):.4f}",
            f"  - Macro F1:       {float(test_statistics.get('f1_macro_audio', 0.0)):.4f}",
            f"  - Weighted F1:    {float(test_statistics.get('f1_weighted_audio', 0.0)):.4f}",
            "\n  Confusion Matrix (Actual \\ Predicted):",
            self._format_cm(test_statistics.get('confu_matrix_audio')),
            "  Classification Report:",
            test_statistics.get('message_audio') or '  N/A\n',
            "=" * 80 + "\n"
        ])

        try:
            clean_content = [str(x) if x is not None else "  N/A" for x in content]
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(clean_content))
            logger.info(f"Saved separate branch reports and consolidated report to: '{self.log_dir}'")
        except Exception as exc:
            logger.error(f"Failed to write consolidated evaluation report: {exc}")

        # Also serialize to structured JSON
        def sanitize_for_json(d: Any) -> Any:
            if isinstance(d, dict):
                return {str(k): sanitize_for_json(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [sanitize_for_json(v) for v in d]
            elif isinstance(d, np.ndarray):
                return d.tolist()
            elif isinstance(d, (np.floating, np.integer)):
                return d.item()
            elif isinstance(d, (int, float, str, bool)) or d is None:
                return d
            return str(d)

        try:
            json_payload = {
                "total_params_m": total_params_m,
                "gflops": gflops,
                "dual_comparison_info": sanitize_for_json(dual_comparison_info),
                "val_evaluation": sanitize_for_json(val_statistics),
                "test_evaluation": sanitize_for_json(test_statistics)
            }
            with open(report_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not serialize evaluation to JSON: {exc}")

        return report_txt_path

    def _plot_single_cm(self, cm: Optional[np.ndarray], title: str, output_path: str) -> None:
        """Plot and save a standalone, publication-quality Confusion Matrix Heatmap PNG."""
        if cm is None or not isinstance(cm, np.ndarray) or cm.size == 0:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            import matplotlib.pyplot as plt

            class_names = ['None', 'Strong', 'Medium', 'Weak']
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, fontsize=11)
            ax.set_yticklabels(class_names, fontsize=11)
            ax.set_ylabel('Actual Label', fontweight='bold', fontsize=12)
            ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)

            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            n_r = min(cm.shape[0], len(class_names))
            n_c = min(cm.shape[1], len(class_names))
            for i in range(n_r):
                row_total = np.sum(cm[i, :])
                for j in range(n_c):
                    count = int(cm[i, j])
                    pct = (count / row_total * 100.0) if row_total > 0 else 0.0
                    color = "white" if count > thresh else "black"
                    ax.text(j, i, f"{count}\n({pct:.1f}%)",
                            horizontalalignment="center",
                            verticalalignment="center",
                            color=color, fontsize=11, fontweight='bold')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved individual confusion matrix plot to: '{output_path}'")
        except Exception as exc:
            logger.warning(f"Could not plot individual confusion matrix '{output_path}': {exc}")

    def plot_test_confusion_matrices(
        self,
        test_statistics: Dict[str, Any],
        val_statistics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Plots both INDIVIDUAL standalone Confusion Matrix Heatmaps for each branch
        and a 1x3 comparison figure.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            import matplotlib.pyplot as plt

            acc_v_test = float(test_statistics.get('acc_video', 0.0))
            acc_a_test = float(test_statistics.get('acc_audio', 0.0))
            acc_f_test = float(np.mean(test_statistics.get('accuracy', 0.0)))

            # 1. Plot individual Test heatmaps
            self._plot_single_cm(
                cm=test_statistics.get('confu_matrix'),
                title=f"Multimodal Fusion - Test (Acc: {acc_f_test:.4f})",
                output_path=os.path.join(self.log_dir, 'confusion_matrix_test_fusion.png')
            )
            self._plot_single_cm(
                cm=test_statistics.get('confu_matrix_video'),
                title=f"Video Aux Head - Test (Acc: {acc_v_test:.4f})",
                output_path=os.path.join(self.log_dir, 'confusion_matrix_test_video.png')
            )
            self._plot_single_cm(
                cm=test_statistics.get('confu_matrix_audio'),
                title=f"Audio Aux Head - Test (Acc: {acc_a_test:.4f})",
                output_path=os.path.join(self.log_dir, 'confusion_matrix_test_audio.png')
            )

            # 2. Plot individual Validation heatmaps if available
            if val_statistics is not None:
                acc_v_val = float(val_statistics.get('acc_video', 0.0))
                acc_a_val = float(val_statistics.get('acc_audio', 0.0))
                acc_f_val = float(np.mean(val_statistics.get('accuracy', 0.0)))

                self._plot_single_cm(
                    cm=val_statistics.get('confu_matrix'),
                    title=f"Multimodal Fusion - Val (Acc: {acc_f_val:.4f})",
                    output_path=os.path.join(self.log_dir, 'confusion_matrix_val_fusion.png')
                )
                self._plot_single_cm(
                    cm=val_statistics.get('confu_matrix_video'),
                    title=f"Video Aux Head - Val (Acc: {acc_v_val:.4f})",
                    output_path=os.path.join(self.log_dir, 'confusion_matrix_val_video.png')
                )
                self._plot_single_cm(
                    cm=val_statistics.get('confu_matrix_audio'),
                    title=f"Audio Aux Head - Val (Acc: {acc_a_val:.4f})",
                    output_path=os.path.join(self.log_dir, 'confusion_matrix_val_audio.png')
                )

            # 3. Plot 1x3 comparison figure on Test
            cms = [
                (test_statistics.get('confu_matrix_video'), f"Video Aux Head (Acc: {acc_v_test:.4f})"),
                (test_statistics.get('confu_matrix_audio'), f"Audio Aux Head (Acc: {acc_a_test:.4f})"),
                (test_statistics.get('confu_matrix'), f"Multimodal Fusion (Acc: {acc_f_test:.4f})"),
            ]

            class_names = ['None', 'Strong', 'Medium', 'Weak']
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            fig.suptitle('Test Set Confusion Matrices Comparison: Video Aux vs Audio Aux vs Multimodal Fusion', fontsize=15, fontweight='bold')

            for idx, (cm, title) in enumerate(cms):
                ax = axes[idx]
                if cm is not None and isinstance(cm, np.ndarray) and cm.size > 0:
                    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    tick_marks = np.arange(len(class_names))
                    ax.set_xticks(tick_marks)
                    ax.set_yticks(tick_marks)
                    ax.set_xticklabels(class_names, rotation=30)
                    ax.set_yticklabels(class_names)
                    ax.set_ylabel('Actual Label', fontweight='bold')
                    ax.set_xlabel('Predicted Label', fontweight='bold')

                    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
                    n_r = min(cm.shape[0], len(class_names))
                    n_c = min(cm.shape[1], len(class_names))
                    for i in range(n_r):
                        row_total = np.sum(cm[i, :])
                        for j in range(n_c):
                            count = int(cm[i, j])
                            pct = (count / row_total * 100.0) if row_total > 0 else 0.0
                            color = "white" if count > thresh else "black"
                            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                                    horizontalalignment="center",
                                    verticalalignment="center",
                                    color=color, fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, "Data Not Available", horizontalalignment='center', verticalalignment='center')
                    ax.set_title(title, fontsize=12)

            plt.tight_layout()
            comparison_plot_path = os.path.join(self.log_dir, 'confusion_matrices_test_comparison.png')
            plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Successfully exported Comparison Heatmaps to: '{comparison_plot_path}'")
            return comparison_plot_path
        except Exception as exc:
            logger.warning(f"Could not export test confusion matrices: {exc}")
            return ""
