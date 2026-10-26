import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    VALID_CACHE_MODES,
    SplitterConfig,
)
from dataset.data_split import FishDataSplitter
from transforms.video_transform import ConsistentVideoTransform


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _decode_video_frames_raw(video_path: str, image_size: int = 224, num_frames: int = 2) -> np.ndarray:
    """
    Decode video frames into uint8 NumPy array [num_frames, image_size, image_size, 3] RGB.
    Uses decord with cv2 fallback.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, width=image_size, height=image_size, ctx=cpu(0))
        total = len(vr)
        if total > 0:
            indices = np.linspace(0, total - 1, num_frames).astype(int).tolist()
            return vr.get_batch(indices).asnumpy()
    except Exception:
        pass

    import cv2
    frames = []
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            indices = np.linspace(0, total - 1, num_frames).astype(int).tolist()
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[0] != image_size or frame.shape[1] != image_size:
                        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                    frames.append(frame)
        cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    return np.stack(frames[:num_frames])


def _decode_audio_waveform_raw(audio_path: str, sample_rate: int = 256000) -> np.ndarray:
    """
    Load raw audio waveform into float32 NumPy array [sample_rate * 2].
    Zero pads or truncates to exactly 2 seconds (512,000 samples @ 256 kHz).
    """
    target_len = sample_rate * 2
    if audio_path and os.path.exists(audio_path):
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            if waveform.ndim == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            y = waveform.squeeze(0).to(torch.float32)
            if y.numel() > target_len:
                y = y[:target_len].clone()
            elif y.numel() < target_len:
                y = torch.nn.functional.pad(y, (0, target_len - y.numel()))
            return y.numpy()
        except Exception:
            pass

        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=-1)
            if sr != sample_rate:
                from scipy.signal import resample_poly
                import math
                gcd = math.gcd(sample_rate, sr)
                data = resample_poly(data, sample_rate // gcd, sr // gcd).astype(np.float32)
            if len(data) > target_len:
                data = data[:target_len]
            elif len(data) < target_len:
                data = np.pad(data, (0, target_len - len(data)))
            return data
        except Exception:
            pass

    return np.zeros(target_len, dtype=np.float32)


def _parse_target_to_onehot(tgt: Any) -> np.ndarray:
    """
    Parses label representation (int index, string class name, or vector)
    into a canonical 4-element one-hot float32 array.
    """
    if isinstance(tgt, (int, np.integer)):
        tgt_onehot = np.zeros(4, dtype=np.float32)
        if 0 <= int(tgt) < 4:
            tgt_onehot[int(tgt)] = 1.0
        return tgt_onehot
    if isinstance(tgt, str):
        tgt_str = tgt.strip().lower()
        class_map = {"none": 0, "strong": 1, "medium": 2, "weak": 3}
        tgt_onehot = np.zeros(4, dtype=np.float32)
        if tgt_str in class_map:
            tgt_onehot[class_map[tgt_str]] = 1.0
        return tgt_onehot
    try:
        arr = np.array(tgt, dtype=np.float32)
        return arr.reshape(4) if arr.size == 4 else np.zeros(4, dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


class FishMultimodalDataLoader:
    """
    Unified DataLoader Manager for Multimodal Fish Feeding Intensity Assessment.
    Loads paired Video Frames [B, T, 3, H, W] and Audio Waveforms [B, 512000].
    Implements multi-threaded RAM preloading for zero disk I/O in epochs 2+.
    """
    def __init__(
        self,
        batch_size: int = 32,
        dataloader_workers: int = -1,
        prefetch_factor: Optional[int] = None,
        cache_mode: str = "ram",
        image_size: int = 224,
        num_frames: int = 2,
        sample_rate: int = 256000,
        splitter_config: Optional[SplitterConfig] = None,
        load_video: bool = True,
        load_audio: bool = True,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.prefetch_factor = prefetch_factor
        self.cache_mode = cache_mode.lower()
        if self.cache_mode not in VALID_CACHE_MODES:
            raise ValueError(f"Invalid cache_mode='{cache_mode}'. Expected one of {sorted(VALID_CACHE_MODES)}.")

        self.image_size = image_size
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.load_video = bool(load_video)
        self.load_audio = bool(load_audio)
        self.persistent_workers = bool(persistent_workers) if persistent_workers is not None else (self.dataloader_workers > 0)

        if self.dataloader_workers == -1:
            max_cpu = os.cpu_count()
            if max_cpu is None or max_cpu <= 0:
                self.dataloader_workers = 0
            elif max_cpu <= 4:
                self.dataloader_workers = 2
            else:
                self.dataloader_workers = (max_cpu // 2) + 1

        self.splitter_config = splitter_config if splitter_config is not None else SplitterConfig(include_video=self.load_video)
        if not self.load_video:
            self.splitter_config.include_video = False

        try:
            self.splitter = FishDataSplitter(config=self.splitter_config)
            self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()
        except Exception as exc:
            logger.error(f"Failed to load dataset from '{self.splitter_config.dataset_path}': {exc}")
            raise

        logger.info("==================================================")
        logger.info("Initializing FishMultimodalDataLoader:")
        logger.info(f"  - Batch Size:               {self.batch_size}")
        logger.info(f"  - DataLoader Workers:       {self.dataloader_workers}")
        logger.info(f"  - Persistent Workers:       {self.persistent_workers}")
        logger.info(f"  - Cache Mode:               {self.cache_mode}")
        logger.info(f"  - Image Resolution:         {self.image_size}x{self.image_size}")
        logger.info(f"  - Number of Frames:         {self.num_frames}")
        logger.info(f"  - Audio Sample Rate:        {self.sample_rate} Hz")
        logger.info("==================================================")

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        clip_names = [data['clip_name'] for data in batch]
        targets = [data['target'] for data in batch]

        if batch and batch[0].get('video_form') is not None and batch[0]['video_form'].numel() > 0:
            video_tensors = torch.stack([data['video_form'] for data in batch])
        else:
            video_tensors = torch.empty(0, dtype=torch.float32)

        if batch and batch[0].get('audio_form') is not None and batch[0]['audio_form'].numel() > 0:
            audio_tensors = torch.stack([data['audio_form'] for data in batch])
        else:
            audio_tensors = torch.empty(0, dtype=torch.float32)

        targets_tensor = torch.FloatTensor(np.array(targets))

        return {
            'clip_name': clip_names,
            'video_form': video_tensors,
            'audio_form': audio_tensors,
            'target': targets_tensor
        }

    class _InnerDataset(Dataset):
        def __init__(self, parent: 'FishMultimodalDataLoader', split: str) -> None:
            self.parent = parent
            self.split = split

            if self.split == 'train':
                self.data_dict = parent.train_dict
            elif self.split == 'test':
                self.data_dict = parent.test_dict
            elif self.split == 'val':
                self.data_dict = parent.val_dict
            else:
                raise ValueError(f"Invalid split value '{self.split}'.")

            transforms = ConsistentVideoTransform.get_transforms(image_size=parent.image_size)
            self.transform = transforms[self.split]
            self.ram_cache = None

            if self.parent.cache_mode == "ram" and len(self.data_dict) > 0:
                self._preload_to_ram()

        def _preload_to_ram(self) -> None:
            import concurrent.futures
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            preload_threads = max(1, self.parent.dataloader_workers * 2 if self.parent.dataloader_workers > 0 else 4)
            modality_str = "Audio-only" if (self.parent.load_audio and not self.parent.load_video) else ("Video-only" if (self.parent.load_video and not self.parent.load_audio) else "Multimodal")
            logger.info("==================================================")
            logger.info(f"Starting {modality_str} -> RAM preload for split '{self.split}' ({len(self.data_dict)} samples)...")
            logger.info(f"Using ThreadPoolExecutor with {preload_threads} workers for RAM preload.")

            def load_single_entry(idx_and_item):
                idx, item = idx_and_item
                if isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        audio_p, video_p, tgt = str(item[0]), str(item[1]), item[2]
                    elif len(item) == 2:
                        p0, tgt = str(item[0]), item[1]
                        if "_audio_" in p0 or p0.endswith(".wav"):
                            audio_p, video_p = p0, ""
                        else:
                            audio_p, video_p = "", p0
                    else:
                        audio_p, video_p, tgt = "", "", 0
                elif isinstance(item, dict):
                    audio_p = str(item.get('audio_path', ''))
                    video_p = str(item.get('video_path', ''))
                    tgt = item.get('label', item.get('target', 0))
                else:
                    audio_p, video_p, tgt = "", "", 0

                if not audio_p and video_p:
                    cand = video_p.replace("/video/", "/audio/").replace("\\video\\", "\\audio\\").replace("_video_", "_audio_")
                    if cand.endswith(".mp4"):
                        cand = cand[:-4] + ".wav"
                    if os.path.exists(cand):
                        audio_p = cand
                if not video_p and audio_p:
                    cand = audio_p.replace("/audio/", "/video/").replace("\\audio\\", "\\video\\").replace("_audio_", "_video_")
                    if cand.endswith(".wav"):
                        cand = cand[:-4] + ".mp4"
                    if os.path.exists(cand):
                        video_p = cand

                tgt_onehot = _parse_target_to_onehot(tgt)

                if self.parent.load_video:
                    video_raw = _decode_video_frames_raw(video_p, self.parent.image_size, self.parent.num_frames)
                else:
                    video_raw = None

                if self.parent.load_audio:
                    audio_raw = _decode_audio_waveform_raw(audio_p, self.parent.sample_rate)
                else:
                    audio_raw = None

                clip_name = os.path.basename(video_p or audio_p or f"sample_{idx}")

                return idx, (video_raw, audio_raw, tgt_onehot, clip_name)

            cache = [None] * len(self.data_dict)
            total_bytes = 0
            indexed_items = list(enumerate(self.data_dict))

            with concurrent.futures.ThreadPoolExecutor(max_workers=preload_threads) as executor:
                futures = {
                    executor.submit(load_single_entry, item): item[0]
                    for item in indexed_items
                }
                iterator = concurrent.futures.as_completed(futures)
                if has_tqdm:
                    pbar = tqdm(total=len(futures), desc=f"Preloading {self.split} ({modality_str}) to RAM")
                    for future in iterator:
                        idx = futures[future]
                        try:
                            result_idx, sample = future.result()
                            cache[result_idx] = sample
                            v_bytes = sample[0].nbytes if sample[0] is not None else 0
                            a_bytes = sample[1].nbytes if sample[1] is not None else 0
                            total_bytes += v_bytes + a_bytes
                        except Exception as exc:
                            logger.error(f"Error preloading sample {idx}: {exc}")
                        pbar.update(1)
                    pbar.close()
                else:
                    for future in iterator:
                        idx = futures[future]
                        try:
                            result_idx, sample = future.result()
                            cache[result_idx] = sample
                            v_bytes = sample[0].nbytes if sample[0] is not None else 0
                            a_bytes = sample[1].nbytes if sample[1] is not None else 0
                            total_bytes += v_bytes + a_bytes
                        except Exception as exc:
                            logger.error(f"Error preloading sample {idx}: {exc}")

            self.ram_cache = cache
            cache_mb = total_bytes / (1024 ** 2)
            logger.info(f"Successfully cached '{self.split}' split to RAM ({modality_str}): {len(self.ram_cache)} samples ({cache_mb:.1f} MB)")
            logger.info("==================================================")

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            if self.ram_cache is not None and self.ram_cache[idx] is not None:
                video_raw, audio_raw, target_onehot, clip_name = self.ram_cache[idx]
                if self.parent.load_video and video_raw is not None:
                    video_tensor = self.transform(video_raw)
                    if video_tensor.ndim == 3:
                        video_tensor = video_tensor.unsqueeze(0)
                    video_tensor = video_tensor[:self.parent.num_frames]
                else:
                    video_tensor = torch.empty(0, dtype=torch.float32)

                if self.parent.load_audio and audio_raw is not None:
                    audio_tensor = torch.from_numpy(audio_raw).to(torch.float32)  # [512000] (2.0s @ 256 kHz)
                else:
                    audio_tensor = torch.empty(0, dtype=torch.float32)

                return {
                    'clip_name': clip_name,
                    'video_form': video_tensor,
                    'audio_form': audio_tensor,
                    'target': target_onehot
                }

            # On-the-fly disk loading fallback
            item = self.data_dict[idx]
            if isinstance(item, (list, tuple)):
                audio_p = str(item[0]) if len(item) > 0 else ""
                video_p = str(item[1]) if len(item) > 1 else ""
                tgt = item[2] if len(item) > 2 else 0
            elif isinstance(item, dict):
                audio_p = str(item.get('audio_path', ''))
                video_p = str(item.get('video_path', ''))
                tgt = item.get('label', item.get('target', 0))
            else:
                audio_p, video_p, tgt = "", "", 0

            if not audio_p and video_p:
                cand = video_p.replace("/video/", "/audio/").replace("\\video\\", "\\audio\\").replace("_video_", "_audio_")
                if cand.endswith(".mp4"):
                    cand = cand[:-4] + ".wav"
                if os.path.exists(cand):
                    audio_p = cand

            tgt_onehot = _parse_target_to_onehot(tgt)

            if self.parent.load_video:
                video_raw = _decode_video_frames_raw(video_p, self.parent.image_size, self.parent.num_frames)
                video_tensor = self.transform(video_raw)
                if video_tensor.ndim == 3:
                    video_tensor = video_tensor.unsqueeze(0)
                video_tensor = video_tensor[:self.parent.num_frames]
            else:
                video_tensor = torch.empty(0, dtype=torch.float32)

            if self.parent.load_audio:
                audio_raw = _decode_audio_waveform_raw(audio_p, self.parent.sample_rate)
                audio_tensor = torch.from_numpy(audio_raw).to(torch.float32)
            else:
                audio_tensor = torch.empty(0, dtype=torch.float32)

            clip_name = os.path.basename(video_p or audio_p or f"sample_{idx}")

            return {
                'clip_name': clip_name,
                'video_form': video_tensor,
                'audio_form': audio_tensor,
                'target': tgt_onehot
            }

    def get_data_loader(self, split: str = 'train') -> DataLoader:
        dataset = self._InnerDataset(self, split)
        is_train = (split == 'train')

        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': is_train,
            'num_workers': self.dataloader_workers,
            'collate_fn': self.collate_fn,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': is_train,
        }
        if self.dataloader_workers > 0:
            kwargs['persistent_workers'] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs['prefetch_factor'] = self.prefetch_factor

        return DataLoader(dataset, **kwargs)

