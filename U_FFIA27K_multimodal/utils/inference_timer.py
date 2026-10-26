import time
import logging
from typing import Tuple
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceTimer:
    """
    Measures multimodal inference latency and throughput with CUDA warmup.
    """
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def measure_latency_per_sample(
        self,
        video_shape: Tuple[int, ...] = (1, 2, 3, 224, 224),
        audio_shape: Tuple[int, ...] = (1, 512000),
        warm_up_steps: int = 10,
        num_steps: int = 50
    ) -> float:
        dummy_video = torch.randn(*video_shape).to(self.device)
        dummy_audio = torch.randn(*audio_shape).to(self.device)
        self.model.eval()

        is_cuda = self.device.type == 'cuda'

        # Warm-up
        with torch.no_grad():
            for _ in range(warm_up_steps):
                _ = self.model(dummy_video, dummy_audio)
                if is_cuda:
                    torch.cuda.synchronize()

        # Measurement
        if is_cuda:
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_steps):
                _ = self.model(dummy_video, dummy_audio)
                if is_cuda:
                    torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time_seconds = end_time - start_time
        latency_per_sample_ms = (total_time_seconds / num_steps) * 1000.0
        throughput_fps = 1000.0 / latency_per_sample_ms

        logger.info("==================================================")
        logger.info("MULTIMODAL INFERENCE PERFORMANCE REPORT:")
        logger.info(f"  - Device:                         {self.device}")
        logger.info(f"  - Avg Inference Latency / sample: {latency_per_sample_ms:.3f} ms")
        logger.info(f"  - Throughput:                     {throughput_fps:.1f} samples/sec")
        logger.info("==================================================")

        return latency_per_sample_ms
