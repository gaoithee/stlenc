import sys
import json
import gc
import torch
import torch.nn.functional as F
import resource
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from utils_measure import CUDATimer
from tokenize_utils import tokenize_only, model_forward_only

import warnings
warnings.filterwarnings("ignore")

MODEL_ID = "saracandu/stlenc-arch-cls"
DEVICE = "cuda"

def get_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

B = int(sys.argv[1])
N = int(sys.argv[2])

print(f"[TRANSFORMER] Starting | B={B}", flush=True)

# ================= DATASET =================
ds = load_dataset(
    "parquet",
    data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet"
)["train"]
formulas = ds["formula"][:B]

# ============================================================
# ================= EMBEDDING: MODEL ALREADY LOADED ==========
# ============================================================
print("[TRANSFORMER] EMBEDDING (model loaded) phase...", flush=True)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device=DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
model.eval()

# warmup (non misurato)
token_batches = tokenize_only(tokenizer, formulas, batch_size=128)
_ = model_forward_only(model, token_batches, DEVICE)
torch.cuda.synchronize()

# misurazione
torch.cuda.synchronize()
with CUDATimer() as t_emb_loaded:
    token_batches = tokenize_only(tokenizer, formulas, batch_size=128)
    z = model_forward_only(model, token_batches, DEVICE)
torch.cuda.synchronize()

VRAM_emb_loaded = torch.cuda.max_memory_allocated(device=DEVICE) / 1024**2
RAM_emb_loaded = get_peak_rss_mb()
T_emb_loaded = t_emb_loaded.elapsed

del z, token_batches
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# ================= SIMILARITY: MODEL ALREADY LOADED ========
# ============================================================
print("[TRANSFORMER] SIMILARITY (model loaded) phase...", flush=True)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device=DEVICE)

torch.cuda.synchronize()
with CUDATimer() as t_sim_loaded:
    token_batches = tokenize_only(tokenizer, formulas, batch_size=128)
    z = model_forward_only(model, token_batches, DEVICE)
    z = F.normalize(z, p=2, dim=1)
    K = z @ z.T
torch.cuda.synchronize()

VRAM_sim_loaded = torch.cuda.max_memory_allocated(device=DEVICE) / 1024**2
RAM_sim_loaded = get_peak_rss_mb()
T_sim_loaded = t_sim_loaded.elapsed

del z, token_batches
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# ================= EMBEDDING: MODEL LOAD INCLUDED ==========
# ============================================================
print("[TRANSFORMER] EMBEDDING (load included) phase...", flush=True)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device=DEVICE)

torch.cuda.synchronize()
with CUDATimer() as t_emb_full:
    tokenizer_full = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_full = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model_full.eval()
    token_batches = tokenize_only(tokenizer_full, formulas, batch_size=128)
    z = model_forward_only(model_full, token_batches, DEVICE)
torch.cuda.synchronize()

VRAM_emb_full = torch.cuda.max_memory_allocated(device=DEVICE) / 1024**2
RAM_emb_full = get_peak_rss_mb()
T_emb_full = t_emb_full.elapsed

del z, token_batches, model_full, tokenizer_full
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# ================= SIMILARITY: MODEL LOAD INCLUDED =========
# ============================================================
print("[TRANSFORMER] SIMILARITY (load included) phase...", flush=True)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device=DEVICE)

torch.cuda.synchronize()
with CUDATimer() as t_sim_full:
    tokenizer_full = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_full = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model_full.eval()
    token_batches = tokenize_only(tokenizer_full, formulas, batch_size=128)
    z = model_forward_only(model_full, token_batches, DEVICE)
    z = F.normalize(z, p=2, dim=1)
    K = z @ z.T
torch.cuda.synchronize()

VRAM_sim_full = torch.cuda.max_memory_allocated(device=DEVICE) / 1024**2
RAM_sim_full = get_peak_rss_mb()
T_sim_full = t_sim_full.elapsed

del z, token_batches, model_full, tokenizer_full
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# ================= RESULTS ================================
# ============================================================

result = {
    # compatibile con runner (modello già caricato)
    "T_Emb": T_emb_loaded,
    "RAM_Emb": RAM_emb_loaded,
    "VRAM_Emb": VRAM_emb_loaded,
    "T_Sim": T_sim_loaded,
    "RAM_Sim": RAM_sim_loaded,
    "VRAM_Sim": VRAM_sim_loaded,

    # caricamento incluso
    "T_Emb_full": T_emb_full,
    "RAM_Emb_full": RAM_emb_full,
    "VRAM_Emb_full": VRAM_emb_full,
    "T_Sim_full": T_sim_full,
    "RAM_Sim_full": RAM_sim_full,
    "VRAM_Sim_full": VRAM_sim_full
}

print("RESULT_JSON:" + json.dumps(result), flush=True)