import time
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ==========================================
# CONFIGURAZIONE E SETUP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "saracandu/stlenc-arch-cls"
RESULTS_DIR = "./results_speedup"
os.makedirs(RESULTS_DIR, exist_ok=True)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_gpu_memory():
    sync()
    mem = torch.cuda.max_memory_allocated(device=DEVICE) / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats(device=DEVICE)
    return mem

# ==========================================
# TRANSFORMER UTILS
# ==========================================
def tokenize_only(tokenizer, formulas, batch_size):
    tokens = []
    for i in range(0, len(formulas), batch_size):
        batch = formulas[i:i+batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        tokens.append(tok)
    return tokens

def model_forward_only(model, token_batches):
    embs = []
    with torch.no_grad():
        for tok in token_batches:
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            out = model(**tok)
            emb = out.pooler_output
            embs.append(emb)
    return torch.cat(embs, dim=0)

# ==========================================
# CLASSE STL KERNEL (VERSIONE CPU-BOUND & FULL-VAL)
# ==========================================
# ==========================================
# CLASSE STL KERNEL (VERSIONE FULL-GPU)
# ==========================================
class StlKernelBenchmarkGPU:
    def __init__(self, measure, normalize=True, exp_kernel=True, sigma2=0.2,
                 samples=1000, varn=3, points=100, boolean=False):
        
        self.device = DEVICE
        self.traj_measure = measure
        self.normalize = normalize
        self.exp_kernel = exp_kernel
        self.sigma2 = sigma2
        self.samples = samples
        self.boolean = boolean
        
        # 🚀 Campionamento direttamente su GPU
        self.signals = measure.sample(
            points=points,
            samples=samples,
            varn=varn
        ).to(self.device)

    def compute_robustness(self, phis):
        n = self.samples
        k = len(phis)

        rhos = torch.zeros((k, n), device=self.device)
        self_kernels = torch.zeros((k, 1), device=self.device)

        for i, phi in enumerate(phis):

            if self.boolean:
                rho_full = phi.boolean(
                    self.signals,
                    evaluate_at_all_times=True
                ).float()
                rho_full[rho_full == 0.0] = -1.0
            else:
                rho_full = phi.quantitative(
                    self.signals,
                    evaluate_at_all_times=True
                )

            # media su asse temporale
            rho = rho_full.mean(dim=-1).flatten()

            self_kernels[i] = torch.dot(rho, rho) / n
            rhos[i, :] = rho

        return rhos, self_kernels

    def compute_kernel(self, rhos1, rhos2, selfk1, selfk2):

        kernel_matrix = torch.matmul(rhos1, rhos2.T) / self.samples

        if self.normalize:
            norm = torch.sqrt(torch.matmul(selfk1, selfk2.T))
            kernel_matrix = kernel_matrix / (norm + 1e-9)

        if self.exp_kernel:
            dist = 2.0 - 2.0 * kernel_matrix
            kernel_matrix = torch.exp(
                -torch.clamp(dist, min=0.0) / (2 * self.sigma2)
            )

        return kernel_matrix
# ==========================================
# SCRIPT DI BENCHMARK
# ==========================================
def run_benchmark(B_list, N_list):
    results = []
    print("Loading dataset and models...")
    ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet")["train"]
    
    # Import locali per dipendenze specifiche
    from train_arch import BaseMeasure, from_string_to_formula
    formulas_all = ds["formula"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model.eval()

    for B in B_list:
        formulas = formulas_all[:B]
        phis = [from_string_to_formula(f) for f in formulas]
        
        for N in N_list:
            print(f"\n>>> TESTING B={B}, N={N}")
            
            # --- KERNEL (GPU) ---
            k_data = {"T_Emb": np.nan, "M_Emb": np.nan,
                    "T_Sim": np.nan, "M_Sim": np.nan,
                    "Status": "Success"}

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)

                kernel = StlKernelBenchmarkGPU(
                    BaseMeasure(device=DEVICE),
                    samples=N,
                    points=N,
                    varn=3
                )

                # Embedding (robustness computation)
                sync()
                t0 = time.perf_counter()
                rhos, selfk = kernel.compute_robustness(phis)
                sync()
                k_data["T_Emb"] = time.perf_counter() - t0
                k_data["M_Emb"] = get_gpu_memory()

                # Similarity
                sync()
                t0 = time.perf_counter()
                _ = kernel.compute_kernel(rhos, rhos, selfk, selfk)
                sync()
                k_data["T_Sim"] = time.perf_counter() - t0
                k_data["M_Sim"] = get_gpu_memory()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    k_data["Status"] = "OOM"
                    torch.cuda.empty_cache()
                else:
                    raise e

            # --- TRANSFORMER (GPU) ---
            tf_data = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                
                # Embedding
                sync()
                t0 = time.perf_counter()
                token_batches = tokenize_only(tokenizer, formulas, 128)
                z = model_forward_only(model, token_batches)
                z = F.normalize(z, p=2, dim=1)
                sync()
                tf_data["T_Emb"], tf_data["M_Emb"] = time.perf_counter()-t0, get_gpu_memory()
                
                # Similarity
                sync()
                t0 = time.perf_counter()
                _ = z @ z.T
                sync()
                tf_data["T_Sim"], tf_data["M_Sim"] = time.perf_counter()-t0, get_gpu_memory()
            except RuntimeError as e:
                if "out of memory" in str(e).lower(): tf_data["Status"] = "OOM"
                else: raise e

            results.append({
                "B": B, "N": N, 
                "K_Status": k_data["Status"], "TF_Status": tf_data["Status"],
                "T_Emb_K": k_data["T_Emb"], "M_Emb_K": k_data["M_Emb"],
                "T_Sim_K": k_data["T_Sim"], "M_Sim_K": k_data["M_Sim"],
                "T_Emb_TF": tf_data["T_Emb"], "M_Emb_TF": tf_data["M_Emb"],
                "T_Sim_TF": tf_data["T_Sim"], "M_Sim_TF": tf_data["M_Sim"]
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Range ridotto per test rapido, poi ripristina i tuoi valori
    B_LIST = [500, 1000, 2000, 4000]
    N_LIST = [500, 1000, 2000, 4000, 8000, 16000]
    
    df_results = run_benchmark(B_LIST, N_LIST)
    df_results.to_csv(os.path.join(RESULTS_DIR, "benchmark_pairwise_final.csv"), index=False)
    print(f"\nBenchmark salvato in {RESULTS_DIR}")