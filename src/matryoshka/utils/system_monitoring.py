import os
import threading
import time
from typing import Optional, Tuple

import psutil


class MemoryLogger:
    def __init__(self, interval: int = 2, logfile: str = None):
        self.interval = interval
        self.logfile = logfile
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._log_memory, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _log_memory(self):
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            mem_mb = process.memory_info().rss / 1024 ** 2
            msg = f"[MEMORY] {mem_mb:.2f} MB"
            print(msg)

            if self.logfile:
                with open(self.logfile, "a") as f:
                    f.write(msg + "\n")

            time.sleep(self.interval)


class MemoryThresholdCalculator:
    def __init__(self, container_memory_gb: Optional[float] = None, safety_factor: float = 0.6):
        self.container_memory_gb = container_memory_gb
        self.safety_factor = safety_factor
        self.available_memory_bytes = self._get_available_memory()

    def _get_available_memory(self) -> int:
        if self.container_memory_gb is not None:
            return int(self.container_memory_gb * 1024**3)

        # Try to detect Docker memory limit
        try:
            # Check cgroup memory limit (Docker containers)
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                cgroup_limit = int(f.read().strip())

            # If cgroup limit is reasonable (not the huge default), use it
            system_memory = psutil.virtual_memory().total
            if cgroup_limit < system_memory * 0.9:  # Container has a real limit
                return cgroup_limit
        except:
            pass

        # Fallback to system memory
        return psutil.virtual_memory().total

    def calculate_memory_usage(self, B: int, D: int) -> dict:
        bytes_per_float64 = 8
        input_memory = B * D * bytes_per_float64
        output_memory = B * D * D * bytes_per_float64
        total_memory = input_memory + output_memory
        peak_memory = total_memory * 1.2

        return {
            'input_memory_mb': input_memory / (1024**2),
            'output_memory_mb': output_memory / (1024**2),
            'total_memory_mb': total_memory / (1024**2),
            'peak_memory_mb': peak_memory / (1024**2),
            'peak_memory_bytes': peak_memory
        }

    def get_threshold_dimensions(self) -> Tuple[int, float]:
        usable_memory = self.available_memory_bytes * self.safety_factor
        # Account for peak memory usage (20% overhead)
        max_peak_memory = usable_memory / 1.2

        # For outer products: peak_memory ~ B*D*8 + B*D^2*8 = B*D*8*(1+D)
        # Solving for maximum B*D²: B*D^2 <= max_peak_memory / (8 * scaling_factor)
        # where scaling_factor accounts for B*D term: ~ 1.01 for typical D values

        max_output_elements = max_peak_memory / (8 * 1.01)  # B*D^2
        threshold_gb = max_peak_memory / (1024**3)

        return int(max_output_elements), threshold_gb

    def should_use_memmap(self, B: int, D: int) -> Tuple[bool, dict]:
        memory_info = self.calculate_memory_usage(B, D)
        max_elements, threshold_gb = self.get_threshold_dimensions()

        # Decision based on peak memory usage
        use_memmap = memory_info['peak_memory_bytes'] > (self.available_memory_bytes * self.safety_factor)

        # Additional info for decision
        memory_info.update({
            'available_memory_gb': self.available_memory_bytes / (1024**3),
            'usable_memory_gb': self.available_memory_bytes * self.safety_factor / (1024**3),
            'threshold_gb': threshold_gb,
            'max_elements': max_elements,
            'current_elements': B * D * D,
            'use_memmap': use_memmap,
            'memory_utilization': memory_info['peak_memory_bytes'] / self.available_memory_bytes
        })

        return use_memmap, memory_info
