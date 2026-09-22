"""Hardware Resource and Profiling Utility.
Spec Section 2 (TASK-011 & TASK-016).
"""

import time
from typing import Any, Dict, Optional

import torch


class ResourceTracker:
    """Tracks training and inference runtime, peak VRAM, and throughput."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> None:
        """Start tracking resources."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.start_time = time.perf_counter()

    def stop(self) -> Dict[str, Any]:
        """Stop tracking and return resource metrics dictionary."""
        self.end_time = time.perf_counter()
        elapsed_sec = (self.end_time - self.start_time) if self.start_time else 0.0

        metrics = {
            "elapsed_seconds": round(elapsed_sec, 3),
            "device": self.device,
            "peak_vram_mb": 0.0,
            "peak_vram_gb": 0.0,
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
        }

        if torch.cuda.is_available() and "cuda" in self.device:
            peak_bytes = torch.cuda.max_memory_allocated()
            reserved_bytes = torch.cuda.max_memory_reserved()
            allocated_bytes = torch.cuda.memory_allocated()

            metrics["peak_vram_mb"] = round(peak_bytes / (1024**2), 2)
            metrics["peak_vram_gb"] = round(peak_bytes / (1024**3), 3)
            metrics["reserved_vram_mb"] = round(reserved_bytes / (1024**2), 2)
            metrics["allocated_vram_mb"] = round(allocated_bytes / (1024**2), 2)

        return metrics

    @staticmethod
    def measure_inference_latency(
        model: Any,
        sample_texts: list,
        src_lang: str,
        tgt_lang: str,
        n_warmup: int = 2,
        n_runs: int = 5,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """Benchmark single-sentence and batched inference latency."""
        if not sample_texts:
            return {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "samples_per_sec": 0.0}

        # Warmup
        for _ in range(n_warmup):
            _ = model.translate(
                sample_texts[:2], src_lang=src_lang, tgt_lang=tgt_lang, **gen_kwargs
            )

        latencies = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.translate(sample_texts, src_lang=src_lang, tgt_lang=tgt_lang, **gen_kwargs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        total_samples = len(sample_texts)
        mean_batch_latency_ms = sum(latencies) / len(latencies)
        mean_per_sample_latency_ms = mean_batch_latency_ms / total_samples
        samples_per_sec = (
            (total_samples / (mean_batch_latency_ms / 1000.0)) if mean_batch_latency_ms > 0 else 0.0
        )

        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        )

        return {
            "total_samples_benchmarked": total_samples,
            "mean_batch_latency_ms": round(mean_batch_latency_ms, 2),
            "mean_per_sample_latency_ms": round(mean_per_sample_latency_ms, 2),
            "samples_per_sec": round(samples_per_sec, 2),
            "peak_inference_vram_mb": round(peak_vram_mb, 2),
        }
