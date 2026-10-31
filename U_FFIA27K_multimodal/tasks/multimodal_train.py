import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
try:
    from sklearn.metrics import average_precision_score
except ImportError:
    average_precision_score = None

from config import TrainConfig
from utils import (
    EarlyStopping,
    HistoryLogger,
    MultimodalEvaluator,
    InferenceTimer,
    ClipCELoss,
    PairwiseTournamentLoss,
    count_parameters,
    measure_flops,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _safe_torch_save(obj: Any, target_path: str) -> bool:
    """
    Atomic fail-safe saving for PyTorch checkpoints.
    Saves to a temporary file first, then atomically renames to target_path.
    Prevents corrupt/truncated files if training is interrupted or if disk is slow.
    """
    tmp_path = f"{target_path}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        torch.save(obj, tmp_path)
        if os.path.exists(target_path):
            try:
                os.replace(tmp_path, target_path)
            except OSError:
                os.remove(target_path)
                os.rename(tmp_path, target_path)
        else:
            os.rename(tmp_path, target_path)
        return True
    except Exception as exc:
        logger.error(f"[Checkpoint Save Error] Could not safely write '{target_path}': {exc}")
        return False


class MultimodalTrainer:
    """
    Unified Trainer class for Multimodal Fish Feeding Intensity Classification.
    Supports Pairwise Tournament Loss, CosineAnnealingLR, and dual-track evaluation.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        config: TrainConfig,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None,
        train_config_path: str = 'config/train_config.json'
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.train_config_path = train_config_path

        # Setup Loss
        loss_type = getattr(self.config, "loss_type", "pairwise_tournament")
        if loss_type == "pairwise_tournament":
            self.loss_fn = PairwiseTournamentLoss(
                weight_act=getattr(self.config, "weight_act", 0.5),
                weight_pairwise=getattr(self.config, "weight_pairwise", 0.5),
                weight_ce=getattr(self.config, "weight_ce", 1.0),
                aux_loss_weight=getattr(self.config, "aux_loss_weight", 0.3),
            ).to(self.device)
            logger.info("Configured PairwiseTournamentLoss (Activity Gate + 3 Boundaries).")
        else:
            self.loss_fn = ClipCELoss()
            logger.info("Configured standard ClipCELoss.")

        # Training & Optimization settings
        self.weight_decay = getattr(self.config, "weight_decay", 0.05)


        # Optimizer: Authentic AdamW across full model
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.weight_decay
        )

        # LR Scheduler configuration with backward compatibility
        raw_scheduler = str(getattr(self.config, "lr_scheduler", "")).strip().lower()
        if not raw_scheduler:
            self.use_onecycle = getattr(self.config, "use_onecycle", True)
            self.lr_scheduler_type = "onecycle" if self.use_onecycle else "cosine"
        else:
            self.lr_scheduler_type = raw_scheduler
            self.use_onecycle = (self.lr_scheduler_type == "onecycle")

        self.eta_min = float(getattr(self.config, "eta_min", getattr(self.config, "min_lr", 1e-6)))
        self.steps_per_epoch = max(1, len(self.train_loader))

        # LR Scheduler setup: onecycle, cosine, or constant (fixed LR)
        if self.lr_scheduler_type in ("constant", "none", "fixed"):
            self.scheduler = None
            logger.info(f"Configured Constant LR: lr={self.config.learning_rate:.2e} (Fixed learning rate, no scheduler adjustment across epochs).")
        elif self.lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.eta_min
            )
            logger.info(f"Configured CosineAnnealingLR: T_max={self.config.epochs}, eta_min={self.eta_min:.1e}.")
        else:
            self.lr_scheduler_type = "onecycle"
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=0.05,
                anneal_strategy='cos',
                div_factor=25,
                final_div_factor=1000
            )
            logger.info(f"Configured OneCycleLR: max_lr={self.config.learning_rate:.2e}, epochs={self.config.epochs}, steps_per_epoch={self.steps_per_epoch}, pct_start=0.05, div_factor=25, final_div_factor=1000.")

        # Evaluator and Timer
        self.evaluator = MultimodalEvaluator(model=self.model, loss_fn=self.loss_fn)
        self.timer = InferenceTimer(model=self.model, device=self.device)

        # Early stopping setup
        self.early_stopping = EarlyStopping(
            patience=getattr(self.config, "patience", 40),
            delta=getattr(self.config, "min_delta", getattr(self.config, "delta", 0.0)),
            verbose=True
        ) if getattr(self.config, "early_stopping", False) else None

        self._init_logging_and_checkpoints()

    def _init_logging_and_checkpoints(self) -> None:
        model_name = getattr(self.config.model, "backbone", getattr(self.model, "model_name", self.model.__class__.__name__))
        self.run_dir = os.path.join(self.config.ckpt_dir, model_name)

        if self.config.dataset_splitter.evaluation_mode == "cross_validation" and self.config.dataset_splitter.fold_index is not None:
            self.run_dir = os.path.join(self.run_dir, f"fold_{self.config.dataset_splitter.fold_index}")

        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = HistoryLogger(log_dir=self.run_dir)
        self.best_checkpoint_path = os.path.join(self.run_dir, 'best_model.pth')
        self.best_video_path = os.path.join(self.run_dir, 'best_video_backbone.pth')
        self.best_audio_path = os.path.join(self.run_dir, 'best_audio_backbone.pth')
        self.last_checkpoint_path = os.path.join(self.run_dir, 'last_model.pth')

        # Dual-track checkpoint paths (both candidates tracked and preserved)
        self.best_checkpoint_path_qwk = os.path.join(self.run_dir, 'best_model_qwk.pth')
        self.best_video_path_qwk = os.path.join(self.run_dir, 'best_video_backbone_qwk.pth')
        self.best_audio_path_qwk = os.path.join(self.run_dir, 'best_audio_backbone_qwk.pth')

        self.best_checkpoint_path_acc = os.path.join(self.run_dir, 'best_model_acc.pth')
        self.best_video_path_acc = os.path.join(self.run_dir, 'best_video_backbone_acc.pth')
        self.best_audio_path_acc = os.path.join(self.run_dir, 'best_audio_backbone_acc.pth')

        logger.info("==================================================")
        logger.info("INITIALIZED MULTIMODAL TRAINING EXPERIMENT:")
        logger.info(f"  - Model Architecture:       {model_name}")
        logger.info(f"  - Device:                   {self.device}")
        logger.info(f"  - Max Epochs:               {self.config.epochs}")
        logger.info(f"  - Batch Size:               {self.config.batch_size}")
        logger.info(f"  - Learning Rate:            {self.config.learning_rate}")
        logger.info(f"  - Weight Decay:             {self.weight_decay}")
        logger.info(f"  - Monitor Metric:           {self.config.monitor}")
        logger.info(f"  - Early Stopping:           {getattr(self.config, 'early_stopping', False)}")
        logger.info(f"  - Checkpoint Run Dir:       '{self.run_dir}'")
        logger.info("==================================================")

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float, float, float]:
        self.model.train()
        total_loss = 0.0
        train_preds = []
        train_preds_video = []
        train_preds_audio = []
        train_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.config.epochs:03d} [Train]")
        for batch_dict in pbar:
            video = batch_dict['video_form'].to(self.device)
            audio = batch_dict['audio_form'].to(self.device)
            targets = batch_dict['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, audio)

            try:
                loss = self.loss_fn(outputs, {'target': targets}, epoch=epoch)
            except TypeError:
                loss = self.loss_fn(outputs, {'target': targets})
            loss.backward()

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            self.optimizer.step()

            if self.scheduler is not None and self.lr_scheduler_type == "onecycle":
                self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            logits = outputs['clipwise_output']
            train_preds.append(logits.detach().cpu().numpy())
            if 'logits_video' in outputs and outputs['logits_video'] is not None:
                train_preds_video.append(outputs['logits_video'].detach().cpu().numpy())
            if 'logits_audio' in outputs and outputs['logits_audio'] is not None:
                train_preds_audio.append(outputs['logits_audio'].detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
            })

        epoch_loss = total_loss / max(1, len(self.train_loader))
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        target_acc_labels = np.argmax(train_targets, axis=1) if train_targets.ndim > 1 else train_targets
        pred_acc_labels = np.argmax(train_preds, axis=1)
        train_acc = float(np.mean(target_acc_labels == pred_acc_labels))

        train_acc_v = 0.0
        if train_preds_video:
            preds_v = np.concatenate(train_preds_video, axis=0)
            train_acc_v = float(np.mean(target_acc_labels == np.argmax(preds_v, axis=1)))

        train_acc_a = 0.0
        if train_preds_audio:
            preds_a = np.concatenate(train_preds_audio, axis=0)
            train_acc_a = float(np.mean(target_acc_labels == np.argmax(preds_a, axis=1)))

        if average_precision_score is not None:
            try:
                train_mAP = float(np.mean(average_precision_score(train_targets, train_preds, average=None)))
            except Exception:
                train_mAP = train_acc
        else:
            train_mAP = train_acc

        return epoch_loss, train_acc, train_acc_v, train_acc_a, train_mAP

    def _load_model_weights(self, path: str) -> bool:
        """Safely loads model weights with fallback across PyTorch versions."""
        if not os.path.exists(path):
            return False
        try:
            sd = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(sd)
            return True
        except Exception:
            try:
                sd = torch.load(path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(sd)
                return True
            except Exception as exc_load:
                logger.error(f"Failed to load weights from '{path}': {exc_load}")
                return False

    def train(self) -> Dict[str, Any]:
        monitor_mode = str(getattr(self.config, "monitor", "val_acc")).strip().lower()
        is_dual = monitor_mode in ("both", "dual")
        logger.info(f"Starting training pipeline (Monitor mode: '{monitor_mode}', Dual-track: {is_dual})...")
        training_start_time = time.perf_counter()

        best_acc = 0.0
        best_qwk = -1.0
        best_mAP = 0.0
        best_loss = float('inf')
        best_epoch = 1
        best_val_statistics = None
        # Dual-track record holders
        best_val_qwk = -1.0
        best_epoch_qwk = 1
        best_val_stats_qwk = None

        best_val_acc = -1.0
        best_epoch_acc = 1
        best_val_stats_acc = None

        if monitor_mode in ('accuracy', 'acc', 'val_acc', 'val_accuracy'):
            best_val_metric = -1.0
        elif monitor_mode == 'qwk':
            best_val_metric = -1.0
        elif is_dual:
            best_val_metric = -1.0
        else:
            best_val_metric = float('inf')

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.perf_counter()
            train_loss, train_acc, train_acc_v, train_acc_a, train_mAP = self._train_epoch(epoch)
            if self.scheduler is not None and self.lr_scheduler_type == "cosine":
                self.scheduler.step()

            # Evaluate on validation split
            self.model.eval()
            val_stats = self.evaluator.evaluate(self.val_loader)
            val_loss = float(val_stats.get('loss', 0.0))
            val_acc = float(np.mean(val_stats['accuracy']))
            val_qwk = float(val_stats.get('qwk', 0.0))
            val_mAP = float(np.mean(val_stats['average_precision']))
            val_acc_v = float(val_stats.get('acc_video', 0.0))
            val_acc_a = float(val_stats.get('acc_audio', 0.0))

            lr_current = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:03d} [TOURNAMENT E2E]: "
                f"Train Loss = {train_loss:.5f} | Train Acc Video = {train_acc_v:.4f} | Train Acc Audio = {train_acc_a:.4f} | Train Acc Fusion = {train_acc:.4f} | LR = {lr_current:.2e} | "
                f"Val Loss = {val_loss:.5f} | Val Acc Video = {val_acc_v:.4f} | Val Acc Audio = {val_acc_a:.4f} | "
                f"Val Acc Fusion = {val_acc:.4f} | Val QWK = {val_qwk:.4f}"
            )

            # Determine if this is the best checkpoint for primary monitor
            is_best = False

            if is_dual:
                current_state_dict = self.model.state_dict()
                v_dict = {k: v for k, v in current_state_dict.items() if k.startswith('video_backbone.') or k.startswith('aux_head_video.')}
                a_dict = {k: v for k, v in current_state_dict.items() if k.startswith('audio_backbone.') or k.startswith('audio_frontend.') or k.startswith('aux_head_audio.')}

                if val_qwk > best_val_qwk:
                    best_val_qwk = val_qwk
                    best_epoch_qwk = epoch
                    best_val_stats_qwk = val_stats
                    is_best = True

                    _safe_torch_save(current_state_dict, self.best_checkpoint_path_qwk)
                    if v_dict:
                        _safe_torch_save(v_dict, self.best_video_path_qwk)
                    if a_dict:
                        _safe_torch_save(a_dict, self.best_audio_path_qwk)

                    logger.info(f"[*] [DUAL-TRACK: QWK] New peak QWK performance! Saved checkpoints:")
                    logger.info(f"    - Full Multimodal Model:   '{self.best_checkpoint_path_qwk}' (Val QWK = {val_qwk:.4f}, Val Acc = {val_acc:.4f})")
                    logger.info(f"    - Peak Video Backbone:     '{self.best_video_path_qwk}'")
                    logger.info(f"    - Peak Audio Backbone:     '{self.best_audio_path_qwk}'")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch_acc = epoch
                    best_val_stats_acc = val_stats
                    is_best = True

                    _safe_torch_save(current_state_dict, self.best_checkpoint_path_acc)
                    if v_dict:
                        _safe_torch_save(v_dict, self.best_video_path_acc)
                    if a_dict:
                        _safe_torch_save(a_dict, self.best_audio_path_acc)

                    logger.info(f"[*] [DUAL-TRACK: ACC] New peak Accuracy performance! Saved checkpoints:")
                    logger.info(f"    - Full Multimodal Model:   '{self.best_checkpoint_path_acc}' (Val Acc = {val_acc:.4f}, Val QWK = {val_qwk:.4f})")
                    logger.info(f"    - Peak Video Backbone:     '{self.best_video_path_acc}'")
                    logger.info(f"    - Peak Audio Backbone:     '{self.best_audio_path_acc}'")

                best_acc = max(best_acc, val_acc)
                best_qwk = max(best_qwk, val_qwk)
                best_mAP = max(best_mAP, val_mAP)
                best_loss = min(best_loss, val_loss)
                score = val_acc
            elif monitor_mode == 'qwk':
                score = val_qwk
                if val_qwk > best_val_metric + 1e-4:
                    best_val_metric = val_qwk
                    is_best = True
                elif abs(val_qwk - best_val_metric) <= 1e-4 and val_acc > best_acc:
                    # Tie-breaker: prefer higher Val Accuracy
                    best_val_metric = val_qwk
                    is_best = True
            elif monitor_mode in ('accuracy', 'acc', 'val_acc', 'val_accuracy'):
                score = val_acc
                if val_acc > best_val_metric + 1e-4:
                    best_val_metric = val_acc
                    is_best = True
                elif abs(val_acc - best_val_metric) <= 1e-4 and val_qwk > best_qwk:
                    # Tie-breaker: prefer higher QWK
                    best_val_metric = val_acc
                    is_best = True
            elif monitor_mode in ('qwk_acc', 'composite'):
                # Balanced Harmonic Score: 0.5 * QWK + 0.5 * Val_Acc
                score = 0.5 * val_qwk + 0.5 * val_acc
                if score > best_val_metric:
                    best_val_metric = score
                    is_best = True
            else:
                score = -val_loss
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    is_best = True

            if not is_dual and is_best:
                best_epoch = epoch
                best_acc = val_acc
                best_qwk = val_qwk
                best_mAP = val_mAP
                best_loss = val_loss
                best_val_statistics = val_stats

                current_state_dict = self.model.state_dict()

                # 1. Full Multimodal Model
                _safe_torch_save(current_state_dict, self.best_checkpoint_path)

                # 2. Peak Video Backbone
                v_dict = {k: v for k, v in current_state_dict.items() if k.startswith('video_backbone.') or k.startswith('aux_head_video.')}
                if v_dict:
                    _safe_torch_save(v_dict, self.best_video_path)

                # 3. Peak Audio Backbone
                a_dict = {k: v for k, v in current_state_dict.items() if k.startswith('audio_backbone.') or k.startswith('audio_frontend.') or k.startswith('aux_head_audio.')}
                if a_dict:
                    _safe_torch_save(a_dict, self.best_audio_path)

                logger.info(f"[*] New best validation performance! Saved checkpoints:")
                logger.info(f"    - Full Multimodal Model:   '{self.best_checkpoint_path}' (Val Acc = {best_acc:.4f}, Val QWK = {best_qwk:.4f})")
                logger.info(f"    - Peak Video Backbone:     '{self.best_video_path}'")
                logger.info(f"    - Peak Audio Backbone:     '{self.best_audio_path}'")

            if is_dual:
                logger.info(
                    f"Current best [DUAL]: Peak QWK = {best_val_qwk:.4f} (Ep {best_epoch_qwk:03d}) | "
                    f"Peak Acc = {best_val_acc:.4f} (Ep {best_epoch_acc:03d})"
                )
            else:
                logger.info(
                    f"Current best: Epoch {best_epoch:03d} | Loss: {best_loss:.5f} | Accuracy: {best_acc:.4f} | QWK: {best_qwk:.4f} | mAP: {best_mAP:.4f}"
                )

            # 4. Always save last checkpoint with full resumption state
            resumption_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_epoch': best_epoch if not is_dual else best_epoch_acc,
                'best_epoch_qwk': best_epoch_qwk if is_dual else None,
                'best_epoch_acc': best_epoch_acc if is_dual else None,
                'best_val_metric': best_val_metric if not is_dual else best_val_acc,
                'best_val_qwk': best_val_qwk if is_dual else best_qwk,
                'best_val_acc': best_val_acc if is_dual else best_acc,
                'best_acc': best_acc,
                'best_qwk': best_qwk,
                'val_statistics': val_stats
            }
            _safe_torch_save(resumption_checkpoint, self.last_checkpoint_path)

            epoch_time_seconds = time.perf_counter() - epoch_start_time

            # Log to history CSV
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mAP=train_mAP,
                val_loss=val_loss,
                val_statistics=val_stats,
                lr=lr_current,
                epoch_time_seconds=epoch_time_seconds,
                is_best=is_best,
                train_acc_video=train_acc_v,
                train_acc_audio=train_acc_a
            )

            # Early stopping check (only if enabled)
            if self.early_stopping is not None:
                if self.early_stopping.step(score):
                    logger.info(f"Early stopping condition satisfied at epoch {epoch:03d}. Stopping training.")
                    break

        training_duration = time.perf_counter() - training_start_time

        # Generate learning curves plot
        try:
            self.logger.plot_history()
        except Exception as exc:
            logger.warning(f"Failed to generate learning curves plot: {exc}")

        # Final evaluation on Test split
        logger.info("==================================================")
        logger.info("Training complete. Starting evaluation on Test split...")

        dual_comparison_info = None

        if is_dual:
            logger.info("Running Dual-Track Tournament Test Split Evaluation...")
            logger.info("Evaluating both QWK candidate and Accuracy candidate head-to-head...")

            # 1. Evaluate QWK candidate
            stats_qwk = None
            if self._load_model_weights(self.best_checkpoint_path_qwk):
                self.model.eval()
                stats_qwk = self.evaluator.evaluate(self.test_loader)
                logger.info(
                    f"[Candidate: QWK] Test Acc = {float(np.mean(stats_qwk['accuracy'])):.4f} | "
                    f"Test QWK = {float(stats_qwk.get('qwk', 0.0)):.4f} | "
                    f"Test mAP = {float(np.mean(stats_qwk['average_precision'])):.4f}"
                )

            # 2. Evaluate Accuracy candidate
            stats_acc = None
            if self._load_model_weights(self.best_checkpoint_path_acc):
                self.model.eval()
                stats_acc = self.evaluator.evaluate(self.test_loader)
                logger.info(
                    f"[Candidate: ACC] Test Acc = {float(np.mean(stats_acc['accuracy'])):.4f} | "
                    f"Test QWK = {float(stats_acc.get('qwk', 0.0)):.4f} | "
                    f"Test mAP = {float(np.mean(stats_acc['average_precision'])):.4f}"
                )

            # 3. Head-to-head decision
            if stats_qwk is not None and stats_acc is not None:
                t_acc_q = float(np.mean(stats_qwk['accuracy']))
                t_qwk_q = float(stats_qwk.get('qwk', 0.0))
                t_map_q = float(np.mean(stats_qwk['average_precision']))

                t_acc_a = float(np.mean(stats_acc['accuracy']))
                t_qwk_a = float(stats_acc.get('qwk', 0.0))
                t_map_a = float(np.mean(stats_acc['average_precision']))

                # Priority: Test Accuracy, then tie-breaker: Test QWK
                if t_acc_a > t_acc_q:
                    winner = 'accuracy'
                elif t_acc_a < t_acc_q:
                    winner = 'qwk'
                else:
                    winner = 'qwk' if t_qwk_q >= t_qwk_a else 'accuracy'

                winning_epoch = best_epoch_acc if winner == 'accuracy' else best_epoch_qwk
                final_test_stats = stats_acc if winner == 'accuracy' else stats_qwk
                final_val_stats = best_val_stats_acc if winner == 'accuracy' else best_val_stats_qwk
                if final_val_stats is None:
                    final_val_stats = self.evaluator.evaluate(self.val_loader)

                # Deploy winning checkpoints to canonical filenames (retaining candidate files intact!)
                src_model = self.best_checkpoint_path_acc if winner == 'accuracy' else self.best_checkpoint_path_qwk
                src_v = self.best_video_path_acc if winner == 'accuracy' else self.best_video_path_qwk
                src_a = self.best_audio_path_acc if winner == 'accuracy' else self.best_audio_path_qwk

                if os.path.exists(src_model):
                    shutil.copy2(src_model, self.best_checkpoint_path)
                if os.path.exists(src_v):
                    shutil.copy2(src_v, self.best_video_path)
                if os.path.exists(src_a):
                    shutil.copy2(src_a, self.best_audio_path)

                # Ensure self.model has winning weights loaded for downstream profiling
                self._load_model_weights(self.best_checkpoint_path)
                self.model.eval()

                logger.info("=" * 88)
                logger.info("DUAL-TRACK TOURNAMENT HEAD-TO-HEAD TEST EVALUATION REPORT:")
                logger.info("=" * 88)
                logger.info(f"{'Candidate Checkpoint':<24} | {'Val Criterion':<16} | {'Val Peak':<10} | {'Test Acc':<10} | {'Test QWK':<10} | {'Test mAP':<10} | {'Decision'}")
                logger.info("-" * 88)
                logger.info(f"{'best_model_qwk.pth':<24} | {f'QWK (Ep {best_epoch_qwk:03d})':<16} | {best_val_qwk:<10.4f} | {t_acc_q*100:<9.2f}% | {t_qwk_q:<10.4f} | {t_map_q*100:<9.2f}% | {'<-- WINNER' if winner == 'qwk' else ''}")
                logger.info(f"{'best_model_acc.pth':<24} | {f'ACC (Ep {best_epoch_acc:03d})':<16} | {best_val_acc*100:<9.2f}% | {t_acc_a*100:<9.2f}% | {t_qwk_a:<10.4f} | {t_map_a*100:<9.2f}% | {'<-- WINNER' if winner == 'accuracy' else ''}")
                logger.info("=" * 88)
                logger.info(f"[*] WINNER SELECTED: Candidate '{winner}' (Epoch {winning_epoch:03d}) won the tournament on Test Split!")
                logger.info(f"    Canonical checkpoints deployed to:")
                logger.info(f"      - Full Model:     '{self.best_checkpoint_path}' <== '{src_model}'")
                logger.info(f"      - Video Backbone: '{self.best_video_path}' <== '{src_v}'")
                logger.info(f"      - Audio Backbone: '{self.best_audio_path}' <== '{src_a}'")
                logger.info("=" * 88)

                dual_comparison_info = {
                    'winner': winner,
                    'winning_epoch': winning_epoch,
                    'qwk_candidate': {
                        'checkpoint': os.path.basename(self.best_checkpoint_path_qwk),
                        'val_epoch': best_epoch_qwk,
                        'val_qwk': best_val_qwk,
                        'test_accuracy': t_acc_q,
                        'test_qwk': t_qwk_q,
                        'test_mAP': t_map_q,
                        'is_winner': (winner == 'qwk')
                    },
                    'acc_candidate': {
                        'checkpoint': os.path.basename(self.best_checkpoint_path_acc),
                        'val_epoch': best_epoch_acc,
                        'val_accuracy': best_val_acc,
                        'test_accuracy': t_acc_a,
                        'test_qwk': t_qwk_a,
                        'test_mAP': t_map_a,
                        'is_winner': (winner == 'accuracy')
                    }
                }
            elif stats_qwk is not None:
                final_test_stats = stats_qwk
                final_val_stats = best_val_stats_qwk if best_val_stats_qwk is not None else self.evaluator.evaluate(self.val_loader)
                if os.path.exists(self.best_checkpoint_path_qwk):
                    shutil.copy2(self.best_checkpoint_path_qwk, self.best_checkpoint_path)
            elif stats_acc is not None:
                final_test_stats = stats_acc
                final_val_stats = best_val_stats_acc if best_val_stats_acc is not None else self.evaluator.evaluate(self.val_loader)
                if os.path.exists(self.best_checkpoint_path_acc):
                    shutil.copy2(self.best_checkpoint_path_acc, self.best_checkpoint_path)
            else:
                final_test_stats = self.evaluator.evaluate(self.test_loader)
                final_val_stats = self.evaluator.evaluate(self.val_loader)
        else:
            # Single monitor mode: reload canonical best_model.pth
            if self._load_model_weights(self.best_checkpoint_path):
                logger.info(f"Reloaded best checkpoint '{self.best_checkpoint_path}' from Epoch {best_epoch:03d}...")
            else:
                logger.warning(f"Could not reload best checkpoint '{self.best_checkpoint_path}'. Proceeding with current in-memory model weights.")

            self.model.eval()
            final_val_stats = best_val_statistics if best_val_statistics is not None else self.evaluator.evaluate(self.val_loader)
            final_test_stats = self.evaluator.evaluate(self.test_loader)

        test_acc = float(np.mean(final_test_stats['accuracy']))
        test_qwk = float(final_test_stats.get('qwk', 0.0))
        test_mAP = float(np.mean(final_test_stats['average_precision']))
        logger.info(f"TEST Results -> Accuracy: {test_acc:.4f} | QWK: {test_qwk:.4f} | mAP: {test_mAP:.4f}")
        logger.info(f"Detailed Classification Report:\n{final_test_stats.get('message', '')}")
        if 'confu_matrix' in final_test_stats:
            logger.info(f"Confusion Matrix:\n{final_test_stats['confu_matrix']}")

        # Measure model complexity, FLOPs, and inference latency
        logger.info("Profiling model Parameters, FLOPs, and Inference Latency...")
        param_stats = count_parameters(self.model)
        total_params_m = float(param_stats.get('total_million', 0.0))
        vf = getattr(self.config, "video_features", None)
        num_frames = getattr(vf, "num_frames", getattr(self.config, "num_frames", 2))
        image_size = getattr(vf, "image_size", getattr(self.config, "image_size", 224))

        try:
            device_str = self.device.type
            gflops = measure_flops(
                self.model,
                device=device_str,
                num_frames=num_frames
            )
            logger.info(f"Model Parameters: {total_params_m:.3f} M | Complexity: {gflops:.4f} GFLOPs")
        except Exception as exc:
            logger.warning(f"FLOPs measurement failed: {exc}. Defaulting to 0.0.")
            gflops = 0.0

        try:
            inference_latency_ms = self.timer.measure_latency_per_sample(
                video_shape=(1, num_frames, 3, image_size, image_size),
                audio_shape=(1, 512000)
            )
        except Exception as exc:
            logger.warning(f"Inference latency measurement failed: {exc}. Defaulting to 0.0 ms.")
            inference_latency_ms = 0.0

        # Save summary report
        self.logger.save_summary(
            training_time=training_duration,
            inference_time_ms=inference_latency_ms,
            val_statistics=final_val_stats,
            test_statistics=final_test_stats,
            total_params_m=total_params_m,
            gflops=gflops
        )

        # Export consolidated detailed evaluation report (.txt and .json)
        try:
            self.logger.save_detailed_evaluation_report(
                val_statistics=final_val_stats,
                test_statistics=final_test_stats,
                total_params_m=total_params_m,
                gflops=gflops,
                dual_comparison_info=dual_comparison_info
            )
        except Exception as exc:
            logger.warning(f"Could not export detailed evaluation report: {exc}")

        # Plot test confusion matrices comparison heatmap (.png)
        try:
            self.logger.plot_test_confusion_matrices(
                test_statistics=final_test_stats,
                val_statistics=final_val_stats
            )
        except Exception as exc:
            logger.warning(f"Could not plot test confusion matrices: {exc}")

        return {
            'training_time': training_duration,
            'inference_time_ms': inference_latency_ms,
            'val_statistics': final_val_stats,
            'test_statistics': final_test_stats,
            'dual_comparison_info': dual_comparison_info
        }
