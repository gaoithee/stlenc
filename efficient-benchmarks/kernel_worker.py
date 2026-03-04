import sys
import json
import gc
import torch
import numpy as np
import resource
from datasets import load_dataset
from utils_measure import CUDATimer
from stl_kernel import StlKernel, BaseMeasure, from_string_to_formula

import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda"

B = int(sys.argv[1])
N = int(sys.argv[2])
P = 1000

def get_peak_rss_mb():
    # ru_maxrss è in KB su Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"[KERNEL] Starting | B={B} | N={N}", flush=True)

# dataset
ds = load_dataset(
    "parquet",
    data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet"
)["train"]

formulas = ds["formula"][:B]
phis = [from_string_to_formula(f) for f in formulas]

torch.cuda.empty_cache()
gc.collect()

# =========================================================
# ================= EMBEDDING =============================
# =========================================================

print("[KERNEL] EMBEDDING phase...", flush=True)

torch.cuda.reset_peak_memory_stats(device=DEVICE)
rss_before = get_peak_rss_mb()

m = BaseMeasure(device="cpu")
s_cpu = m.sample(points=P, samples=N, varn=3)
s_gpu = s_cpu.to(DEVICE)

k = StlKernel(
    measure=BaseMeasure(device=DEVICE),
    signals=s_gpu,
    samples=N,
    points=P,
    varn=3,
    integrate_time=True
)

torch.cuda.synchronize()
with CUDATimer() as t:
    rhos, selfk, lengths = k._compute_robustness_time(phis)
torch.cuda.synchronize()

rss_after = get_peak_rss_mb()
vram_peak = torch.cuda.max_memory_allocated(device=DEVICE) / 1024**2

T_emb = t.elapsed
RAM_emb = rss_after          # peak reale del processo
VRAM_emb = vram_peak         # peak GPU reale

print(
    f"[KERNEL] EMBEDDING done | "
    f"T={T_emb:.2f}s | RAM={RAM_emb:.0f}MB | VRAM={VRAM_emb:.0f}MB",
    flush=True
)

del rhos, selfk, lengths, k, s_gpu
gc.collect()
torch.cuda.empty_cache()

# =========================================================
# ================= SIMILARITY ============================
# =========================================================

print("[KERNEL] SIMILARITY phase...", flush=True)

torch.cuda.reset_peak_memory_stats(device=DEVICE)
rss_before = get_peak_rss_mb()

m = BaseMeasure(device="cpu")
s_cpu = m.sample(points=P, samples=N, varn=3)
s_gpu = s_cpu.to(DEVICE)

k = StlKernel(
    measure=BaseMeasure(device=DEVICE),
    signals=s_gpu,
    samples=N,
    points=P,
    varn=3,
    integrate_time=True
)

torch.cuda.synchronize()
with CUDATimer() as t:
    K = k.compute_bag_bag(phis, phis)
torch.cuda.synchronize()

rss_after = get_peak_rss_mb()
vram_peak = torch.cuda.max_memory_allocated(device=DEVICE) / 1024**2

T_sim = t.elapsed
RAM_sim = rss_after
VRAM_sim = vram_peak

print(
    f"[KERNEL] SIMILARITY done | "
    f"T={T_sim:.2f}s | RAM={RAM_sim:.0f}MB | VRAM={VRAM_sim:.0f}MB",
    flush=True
)

result = {
    "T_Emb": T_emb,
    "RAM_Emb": RAM_emb,
    "VRAM_Emb": VRAM_emb,
    "T_Sim": T_sim,
    "RAM_Sim": RAM_sim,
    "VRAM_Sim": VRAM_sim,
}

print("RESULT_JSON:" + json.dumps(result), flush=True)