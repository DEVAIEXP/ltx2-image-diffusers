import ctypes
import gc
import importlib.metadata
import os
from pathlib import Path
import threading
import time

import psutil
import torch


def is_wsl() -> bool:
    if os.name == "nt":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def should_malloc_trim() -> bool:
    value = os.environ.get("LTX_IMAGE_MALLOC_TRIM")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return not is_wsl()


def get_sdnq_version():
    try:
        return importlib.metadata.version("sdnq")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        try:
            import sdnq
        except Exception:
            return "unavailable"
        return getattr(sdnq, "__version__", "unknown")


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if should_malloc_trim():
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def get_ram_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024**2
    except FileNotFoundError:
        pass
    return psutil.Process().memory_full_info().uss / 1024**3


def get_gpu_used_gb(device="cuda:0"):
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024**3


class PeakMemoryMonitor:
    def __init__(self, device="cuda:0", interval=0.1):
        self.device = device
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self._running = False
        self._thread = None
        self._interval = interval

    def start(self):
        self.peak_ram = get_ram_gb()
        self.peak_vram = get_gpu_used_gb(self.device)
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self.peak_ram = max(self.peak_ram, get_ram_gb())
            self.peak_vram = max(self.peak_vram, get_gpu_used_gb(self.device))
            time.sleep(self._interval)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        return self.peak_vram, self.peak_ram


class RunTracker:
    def __init__(self, device, run_metrics, interval=0.1, show_metrics=True):
        self.device = device
        self.run_metrics = run_metrics
        self.show_metrics = show_metrics
        self.monitor = PeakMemoryMonitor(device, interval=interval)
        torch.cuda.init()
        self.vram_baseline = get_gpu_used_gb(device)
        self.script_start = time.time()
        self.global_peak_vram = 0.0
        self.global_peak_ram = 0.0
        if self.show_metrics:
            print(f"VRAM baseline: {self.vram_baseline:.2f} GB")

    def record_event(self, name, elapsed, **metadata):
        event = {"name": name, "elapsed_sec": round(elapsed, 4)}
        event.update(metadata)
        self.run_metrics["events"].append(event)
        if self.show_metrics:
            print(f"  [event] {name}: {elapsed:.4f}s")
        return event

    def step_start(self, step_name):
        self.monitor.start()
        print(f"\n{'-' * 70}")
        print(f"  {step_name}")
        print(f"{'-' * 70}")
        return time.time()

    def step_end(self, step_name, t0):
        elapsed = time.time() - t0
        peak_vram_abs, peak_ram = self.monitor.stop()
        peak_vram = peak_vram_abs - self.vram_baseline
        if self.show_metrics:
            print(f"  [{step_name}] {elapsed:.1f}s | Peak VRAM: {peak_vram:.2f} GB | Peak RAM: {peak_ram:.2f} GB")
        self.global_peak_vram = max(self.global_peak_vram, peak_vram)
        self.global_peak_ram = max(self.global_peak_ram, peak_ram)
        metric = {
            "name": step_name,
            "elapsed_sec": round(elapsed, 4),
            "peak_vram_gb": round(peak_vram, 4),
            "peak_ram_gb": round(peak_ram, 4),
        }
        self.run_metrics["steps"].append(metric)
        return metric

    def total_elapsed(self):
        return time.time() - self.script_start
