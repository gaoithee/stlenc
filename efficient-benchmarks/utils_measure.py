import os
import time
import psutil
import torch
import subprocess

# ---- CPU RAM ----
def get_cpu_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)

# ---- REAL GPU VRAM (non PyTorch allocator!) ----
def get_gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(out.decode().strip())
    except:
        return -1

# ---- accurate cuda timing ----
class CUDATimer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.t1 = time.perf_counter()
        self.elapsed = self.t1 - self.t0