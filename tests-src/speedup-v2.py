import time
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import psutil


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
    # Restituisce il picco di memoria allocata dall'ultimo reset
    mem = torch.cuda.max_memory_allocated(device=DEVICE) / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats(device=DEVICE)
    return mem

def get_cpu_ram_usage():
    # Restituisce la memoria RAM usata dal processo corrente in MB
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

# ==========================================
# TRANSFORMER UTILS
# ==========================================
def tokenize_only(tokenizer, formulas, batch_size=128):
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

class StlKernel:
    def __init__(
        self,
        measure,
        normalize=True,
        exp_kernel=True,
        sigma2=0.2, # 0.5 meglio, inizialmente era a 0.2
        integrate_time=False,
        samples=100000,
        varn=2,
        points=1000,
        boolean=False,
        signals=None,
    ):
        self.traj_measure = measure
        self.exp_kernel = exp_kernel
        self.normalize = normalize
        self.sigma2 = sigma2
        self.samples = samples
        self.varn = varn
        self.points = points
        self.integrate_time = integrate_time
        if signals is not None:
            self.signals = signals
        else:
            self.signals = measure.sample(points=points, samples=samples, varn=varn)
        self.boolean = boolean

    def compute(self, phi1, phi2):
        return self.compute_one_one(phi1, phi2)

    def compute_one_one(self, phi1, phi2):
        phis1: list = [phi1]
        phis2: list = [phi2]
        ker = self.compute_bag_bag(phis1, phis2)
        return ker[0, 0]

    def compute_bag(self, phis, return_robustness=True):
        if self.integrate_time:
            rhos, selfk, len0 = self._compute_robustness_time(phis)
            kernel_matrix = self._compute_kernel_time(
                rhos, rhos, selfk, selfk, len0, len0
            )
        else:
            rhos, selfk = self._compute_robustness_no_time(phis)
            kernel_matrix = self._compute_kernel_no_time(rhos, rhos, selfk, selfk)
            len0 = None
        if return_robustness:
            return kernel_matrix.cpu(), rhos, selfk, len0
        else:
            return kernel_matrix.cpu()

    def compute_one_bag(self, phi1, phis2, return_robustness=False):
        phis1: list = [phi1]
        return self.compute_bag_bag(phis1, phis2, return_robustness)

    def compute_bag_bag(self, phis1, phis2, return_robustness=False):
        if self.integrate_time:
            rhos1, selfk1, len1 = self._compute_robustness_time(phis1)
            rhos2, selfk2, len2 = self._compute_robustness_time(phis2)
            kernel_matrix = self._compute_kernel_time(
                rhos1, rhos2, selfk1, selfk2, len1, len2
            )
        else:
            rhos1, selfk1 = self._compute_robustness_no_time(phis1)
            rhos2, selfk2 = self._compute_robustness_no_time(phis2)
            len1, len2 = [None, None]
            kernel_matrix = self._compute_kernel_no_time(rhos1, rhos2, selfk1, selfk2)
        if return_robustness:
            return kernel_matrix.cpu(), rhos1, rhos2, selfk1, selfk2, len1, len2
        else:
            return kernel_matrix.cpu()

    def compute_one_from_robustness(self, phi, rhos, rho_self, lengths=None, return_robustness=False):
        phis: list = [phi]
        return self.compute_bag_from_robustness(phis, rhos, rho_self, lengths, return_robustness)

    def compute_bag_from_robustness(self, phis, rhos, rho_self, lengths=None, return_robustness=False):
        if self.integrate_time:
            rhos1, selfk1, len1 = self._compute_robustness_time(phis)
            kernel_matrix = self._compute_kernel_time(
                rhos1, rhos, selfk1, rho_self, len1, lengths
            )
        else:
            rhos1, selfk1 = self._compute_robustness_no_time(phis)
            len1 = None
            kernel_matrix = self._compute_kernel_no_time(rhos1, rhos, selfk1, rho_self)
        if return_robustness:
            return kernel_matrix.cpu(), rhos1, selfk1, len1
        else:
            return kernel_matrix.cpu()

    def _compute_robustness_time(self, phis):
        n = self.samples
        p = self.points
        k = len(phis)
        rhos = torch.zeros((k, n, p), device="cpu")
        lengths = torch.zeros(k)
        self_kernels = torch.zeros((k, 1))
        for i, phi in enumerate(phis):
            if self.boolean:
                rho = phi.boolean(self.signals, evaluate_at_all_times=True).float()
                rho[rho == 0.0] = -1.0
            else:
                rho = phi.quantitative(self.signals, evaluate_at_all_times=True)
            actual_p = rho.size()[2]
            rho = rho.reshape(n, actual_p).cpu()
            rhos[i, :, :actual_p] = rho
            lengths[i] = actual_p
            self_kernels[i] = torch.tensordot(
                rho.reshape(1, n, -1), rho.reshape(1, n, -1), dims=[[1, 2], [1, 2]]
            ) / (actual_p * n)
        return rhos, self_kernels, lengths

    def _compute_robustness_no_time(self, phis):
        n = self.samples
        k = len(phis)
        rhos = torch.zeros((k, n), device=self.traj_measure.device)
        self_kernels = torch.zeros((k, 1), device=self.traj_measure.device)
        for i, phi in enumerate(phis):
            if self.boolean:
                rho = phi.boolean(self.signals, evaluate_at_all_times=False).float()
                rho[rho == 0.0] = -1.0
            else:
                rho = phi.quantitative(self.signals, evaluate_at_all_times=False)
            self_kernels[i] = rho.dot(rho) / n
            rhos[i, :] = rho
        return rhos, self_kernels

    def _compute_kernel_time(self, rhos1, rhos2, selfk1, selfk2, len1, len2):
        kernel_matrix = torch.tensordot(rhos1, rhos2, [[1, 2], [1, 2]])
        length_normalizer = self._compute_trajectory_length_normalizer(len1, len2)
        kernel_matrix = kernel_matrix * length_normalizer / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    def _compute_kernel_no_time(self, rhos1, rhos2, selfk1, selfk2):
        kernel_matrix = torch.tensordot(rhos1, rhos2, [[1], [1]])
        kernel_matrix = kernel_matrix / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    @staticmethod
    def _normalize(kernel_matrix, selfk1, selfk2):
        normalize = torch.sqrt(torch.matmul(selfk1, torch.transpose(selfk2, 0, 1)))
        kernel_matrix = kernel_matrix / normalize
        return kernel_matrix

    def _exponentiate(self, kernel_matrix, selfk1, selfk2, sigma2=None):
        if sigma2 is None:
            sigma2 = self.sigma2
        if self.normalize:
            # selfk is (1.0^2 + 1.0^2)
            selfk = 2.0
        else:
            k1 = selfk1.size()[0]
            k2 = selfk2.size()[0]
            selfk = (selfk1 * selfk1).repeat(1, k2) + torch.transpose(
                selfk2 * selfk2, 0, 1
            ).repeat(k1, 1)
        return torch.exp(-(selfk - 2 * kernel_matrix) / (2 * sigma2))

    @staticmethod
    def _compute_trajectory_length_normalizer(len1, len2):
        k1 = len1.size()[0]
        k2 = len2.size()[0]
        y1 = len1.reshape(-1, 1)
        y1 = y1.repeat(1, k2)
        y2 = len2.repeat(k1, 1)
        return 1.0 / torch.min(y1, y2)


class GramMatrix:
    def __init__(self, kernel, formulae, store_robustness=True, sample=False, sampler=None, bag_size=None):
        self.kernel = kernel
        self.formulae_list = formulae
        # if kernel is computed from robustness at time zero only,
        # we store the robustness for each formula and each sample
        # to speed up computation later
        self.store_robustness = store_robustness
        self.dim = len(self.formulae_list) if not bag_size else int(bag_size)
        self.sample = sample  # whether to generate formulae in a controlled manner
        if self.sample:
            self.t = 0.99 if self.kernel.boolean else 0.85
        self.sampler = sampler  # stl formulae generator
        self._compute_gram_matrix()

    def _compute_gram_matrix(self):
        if self.sample:
            gram = torch.zeros(self.dim, self.dim)
            rhos = torch.zeros((self.dim, self.kernel.samples), device=self.kernel.traj_measure.device) if \
                not self.kernel.integrate_time else torch.zeros((self.dim, self.kernel.samples, self.kernel.points),
                                                                device=self.kernel.traj_measure.device)
            lengths = torch.zeros(self.dim) if self.kernel.integrate_time else np.zeros(self.dim)
            kernels = torch.zeros((self.dim, 1), device=self.kernel.traj_measure.device)
            phis = [self.sampler.sample(nvars=self.kernel.varn)]
            gram[0, :1], rhos[0], kernels[0, :], lengths[0] = self.kernel.compute_bag(phis, return_robustness=True)
            while len(phis) < self.dim:
                i = len(phis)
                phi = self.sampler.sample(nvars=self.kernel.varn)
                gram[i, :i], rhos[i], kernels[i, :], lengths[i] = self.kernel.compute_one_from_robustness(
                    phi, rhos[:i, :], kernels[:i, :], lengths[:i], return_robustness=True)
                if torch.sum(gram[i, :i + 1] >= self.t) < 3:
                    phis.append(phi)
                    gram[:i, i] = gram[i, :i]
                    gram[i, i] = kernels[i, :]

            self.formulae_list = phis
            self.gram = gram.cpu()
            self.robustness = rhos if self.store_robustness else None
            self.self_kernels = kernels if self.store_robustness else None
            self.robustness_lengths = lengths if self.store_robustness else None
        else:
            if self.store_robustness:
                k_matrix, rhos, selfk, len0 = self.kernel.compute_bag(
                    self.formulae_list, return_robustness=True
                )
                self.gram = k_matrix
                self.robustness = rhos
                self.self_kernels = selfk
                self.robustness_lengths = len0
            else:
                self.gram = self.kernel.compute_bag(
                    self.formulae_list, return_robustness=False
                )
                self.robustness = None
                self.self_kernels = None
                self.robustness_lengths = None

    def compute_kernel_vector(self, phi):
        if self.store_robustness:
            return self.kernel.compute_one_from_robustness(
                phi, self.robustness, self.self_kernels, self.robustness_lengths
            )
        else:
            return self.kernel.compute_one_bag(phi, self.formulae_list)

    def compute_bag_kernel_vector(self, phis, generate_phis=False, bag_size=None):
        if generate_phis:
            gram_test = torch.zeros(bag_size, self.dim)  # self.dim, bag_size
            rhos_test = torch.zeros((bag_size, self.kernel.samples), device=self.kernel.traj_measure.device) if \
                not self.kernel.integrate_time else torch.zeros((bag_size, self.kernel.samples, self.kernel.points),
                                                                device=self.kernel.traj_measure.device)
            lengths_test = torch.zeros(bag_size) if self.kernel.integrate_time else np.zeros(bag_size)
            kernels_test = torch.zeros((bag_size, 1), device=self.kernel.traj_measure.device)
            phi_test = []
            while len(phi_test) < bag_size:
                i = len(phi_test)
                phi = self.sampler.sample(nvars=self.kernel.varn)
                if self.store_robustness:
                    gram_test[i, :], rhos_test[i], kernels_test[i, :], lengths_test[i] = \
                        self.kernel.compute_one_from_robustness(phi, self.robustness, self.self_kernels,
                                                                self.robustness_lengths, return_robustness=True)
                else:
                    gram_test[i, :], rhos_test[i], _, kernels_test[i, :], _, lengths_test[i], _ = \
                        self.kernel.compute_one_bag(phi, self.formulae_list, return_robustness=True)
                if not ((rhos_test[i] > 0).all() or (rhos_test[i] < 0).all()):
                    phi_test.append(phi)
            return phi_test, gram_test.cpu()
        else:
            if self.store_robustness:
                return self.kernel.compute_bag_from_robustness(
                    phis, self.robustness, self.self_kernels, self.robustness_lengths
                )
            else:
                return self.kernel.compute_bag_bag(phis, self.formulae_list)

    def invert_regularized(self, alpha):
        regularizer = abs(pow(10, alpha)) * torch.eye(self.dim)
        return torch.inverse(self.gram + regularizer)


# # ==========================================
# def run_benchmark_live(B_list, N_list):
#     results = []
#     print("Loading dataset and models...")
#     ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet")["train"]
    
#     from train_arch import BaseMeasure, from_string_to_formula
#     formulas_all = ds["formula"]

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
#     model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
#     model.eval()

#     print("\n" + "="*105)
#     print(f"{'B':>4} | {'N':>6} | {'Phase':>8} | {'Kernel (Orig)':>15} | {'Transformer':>15} | {'Mem K':>7} | {'Mem TF':>7}")
#     print("-" * 105)

#     for B in B_list:
#         formulas = formulas_all[:B]
#         # Parsing STL (necessario per il Kernel)
#         phis = [from_string_to_formula(f) for f in formulas]
        
#         for N in N_list:
#             P = 100
#             k_data = {"T_Emb": 0, "M_Emb": 0, "T_Sim": 0, "M_Sim": 0, "Status": "Success"}
#             tf_data = {"T_Emb": 0, "M_Emb": 0, "T_Sim": 0, "M_Sim": 0, "Status": "Success"}

#             try:
#                 # --- SETUP AMBIENTE ---
#                 measure = BaseMeasure(device=DEVICE)
#                 signals = measure.sample(points=P, samples=N, varn=3)
#                 # Inizializziamo la classe originale con i segnali pronti
#                 kernel = StlKernel(measure=measure, signals=signals, samples=N, points=P, varn=3, integrate_time=True)

#                 # --- PHASE 1: EMBEDDING ---
#                 # Kernel: compute robustness sui segnali
#                 torch.cuda.empty_cache()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE)
#                 sync()
#                 t0 = time.perf_counter()
#                 rhos, selfk, lengths = kernel._compute_robustness_time(phis)
#                 sync()
#                 k_data["T_Emb"], k_data["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

#                 # Transformer: Encoder(elemento)
#                 torch.cuda.empty_cache()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE)
#                 sync()
#                 t0 = time.perf_counter()
#                 token_batches = tokenize_only(tokenizer, formulas, 128)
#                 z = model_forward_only(model, token_batches)
#                 z = F.normalize(z, p=2, dim=1) # Embedding normalizzato
#                 sync()
#                 tf_data["T_Emb"], tf_data["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

#                 print(f"{B:4} | {N:6} | EMBEDD  | {k_data['T_Emb']:13.2f}s | {tf_data['T_Emb']:13.2f}s | {k_data['M_Emb']:5.0f}MB | {tf_data['M_Emb']:5.0f}MB")

#                 # --- PHASE 2: SIMILARITY ---
#                 # Kernel: compute_bag_bag (pairwise completo tra phis e phis)
#                 torch.cuda.empty_cache()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE)
#                 sync()
#                 t0 = time.perf_counter()
#                 _ = kernel.compute_bag_bag(phis, phis)
#                 sync()
#                 k_data["T_Sim"], k_data["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

#                 # Transformer: cosine similarity (z @ z.T)
#                 torch.cuda.empty_cache()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE)
#                 sync()
#                 t0 = time.perf_counter()
#                 _ = z @ z.T
#                 sync()
#                 tf_data["T_Sim"], tf_data["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

#                 print(f"{'':>4} | {'':>6} | SIMILAR | {k_data['T_Sim']:13.4f}s | {tf_data['T_Sim']:13.4f}s | {k_data['M_Sim']:5.0f}MB | {tf_data['M_Sim']:5.0f}MB")
#                 print("-" * 105)

#             except RuntimeError as e:
#                 if "out of memory" in str(e).lower():
#                     print(f"{B:4} | {N:6} | ERROR   | {'OUT OF MEMORY':^33} |")
#                     k_data["Status"] = "OOM"
#                 else: raise e
#             except Exception as e:
#                 print(f"Kernel Failed: {e}")
#                 k_data["Status"] = "Error"

#             results.append({
#                 "B": B, "N": N, "K_Status": k_data["Status"],
#                 "T_Emb_K": k_data["T_Emb"], "M_Emb_K": k_data["M_Emb"],
#                 "T_Sim_K": k_data["T_Sim"], "M_Sim_K": k_data["M_Sim"],
#                 "T_Emb_TF": tf_data["T_Emb"], "M_Emb_TF": tf_data["M_Emb"],
#                 "T_Sim_TF": tf_data["T_Sim"], "M_Sim_TF": tf_data["M_Sim"]
#             })
            
#     return pd.DataFrame(results)

# if __name__ == "__main__":
#     B_LIST = [500, 1000, 2000, 4000]
#     N_LIST = [500, 1000, 2000, 4000, 8000, 16000]
    
#     df_results = run_benchmark_live(B_LIST, N_LIST)
    
#     # Calcolo Speedup Totale (Emb + Sim) per riga
#     df_results["Speedup_Total"] = (df_results["T_Emb_K"] + df_results["T_Sim_K"]) / \
#                                  (df_results["T_Emb_TF"] + df_results["T_Sim_TF"])
                                 
#     df_results.to_csv(os.path.join(RESULTS_DIR, "benchmark_final_scientific.csv"), index=False)


def run_benchmark_live_oom(B_list, N_list):
    results = []
    print(f"🚀 Inizio Benchmark Scientifico (Cold VRAM) su {DEVICE}...")
    
    ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet")["train"]
    from train_arch import BaseMeasure, from_string_to_formula
    formulas_all = ds["formula"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model.eval()

    print("\n" + "="*110)
    print(f"{'B':>4} | {'N':>6} | {'Phase':>8} | {'Kernel (Orig)':>15} | {'Transformer':>15} | {'Mem K':>7} | {'Mem TF':>7}")
    print("-" * 110)

    for B in B_list:
        formulas = formulas_all[:B]
        phis = [from_string_to_formula(f) for f in formulas]
        
        for N in N_list:
            P = 1000
            k_data = {"T_Emb": 0, "M_Emb": 0, "T_Sim": 0, "M_Sim": 0, "Status": "Success"}
            tf_data = {"T_Emb": 0, "M_Emb": 0, "T_Sim": 0, "M_Sim": 0, "Status": "Success"}

            try:
                # ==========================================================
                # PHASE 1: EMBEDDING (FAIR COLD START)
                # ==========================================================
                
                # --- KERNEL EMBEDDING ---
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                m1 = BaseMeasure(device=DEVICE)
                s1 = m1.sample(points=P, samples=N, varn=3)
                k1 = StlKernel(measure=m1, signals=s1, samples=N, points=P, varn=3, integrate_time=True)
                
                sync()
                t0 = time.perf_counter()
                _ = k1._compute_robustness_time(phis) 
                sync()
                k_data["T_Emb"], k_data["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

                # --- TRANSFORMER EMBEDDING ---
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                sync()
                t0 = time.perf_counter()
                token_batches = tokenize_only(tokenizer, formulas, 128)
                _ = model_forward_only(model, token_batches)
                sync()
                tf_data["T_Emb"], tf_data["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

                print(f"{B:4} | {N:6} | EMBEDD  | {k_data['T_Emb']:13.2f}s | {tf_data['T_Emb']:13.2f}s | {k_data['M_Emb']:5.0f}MB | {tf_data['M_Emb']:5.0f}MB")

                # ==========================================================
                # PHASE 2: SIMILARITY (PAIRWISE COLD START)
                # ==========================================================
                
                # --- KERNEL SIMILARITY (compute_bag_bag) ---
                # Forza ricalcolo su un NUOVO campionamento indipendente
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                m2 = BaseMeasure(device=DEVICE)
                s2 = m2.sample(points=P, samples=N, varn=3)
                k2 = StlKernel(measure=m2, signals=s2, samples=N, points=P, varn=3, integrate_time=True)
                
                sync()
                t0 = time.perf_counter()
                _ = k2.compute_bag_bag(phis, phis) # Calcolo completo O(B*N*P)
                sync()
                k_data["T_Sim"], k_data["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

                # --- TRANSFORMER SIMILARITY (Encoding + Comparison) ---
                # Forza ricalcolo dell'encoding (Cold start per la similarità)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                sync()
                t0 = time.perf_counter()
                
                # 1. Re-Encoding
                token_batches_2 = tokenize_only(tokenizer, formulas, 128)
                z = model_forward_only(model, token_batches_2)
                z = F.normalize(z, p=2, dim=1)
                # 2. Comparison
                _ = z @ z.T 
                
                sync()
                tf_data["T_Sim"], tf_data["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

                print(f"{'':>4} | {'':>6} | SIMILAR | {k_data['T_Sim']:13.2f}s | {tf_data['T_Sim']:13.2f}s | {k_data['M_Sim']:5.0f}MB | {tf_data['M_Sim']:5.0f}MB")
                print("-" * 110)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{B:4} | {N:6} | ERROR   | {'OUT OF MEMORY':^33} |")
                    k_data["Status"] = "OOM"
                    torch.cuda.empty_cache()
                else: raise e
            except Exception as e:
                print(f"Failed @ B={B}, N={N}: {e}")
                k_data["Status"] = "Error"

            results.append({
                "B": B, "N": N, "K_Status": k_data["Status"],
                "T_Emb_K": k_data["T_Emb"], "M_Emb_K": k_data["M_Emb"],
                "T_Sim_K": k_data["T_Sim"], "M_Sim_K": k_data["M_Sim"],
                "T_Emb_TF": tf_data["T_Emb"], "M_Emb_TF": tf_data["M_Emb"],
                "T_Sim_TF": tf_data["T_Sim"], "M_Sim_TF": tf_data["M_Sim"]
            })
            
    return pd.DataFrame(results)

def run_benchmark_live(B_list, N_list):
    results = []
    output_file = os.path.join(RESULTS_DIR, "benchmark_final_cold_vram.csv")
    
    print(f"🚀 Inizio Benchmark Scientifico (Isolamento Totale) su {DEVICE}...")
    
    ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet")["train"]
    from train_arch import BaseMeasure, from_string_to_formula
    formulas_all = ds["formula"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model.eval()

    print("\n" + "="*115)
    print(f"{'B':>4} | {'N':>6} | {'Phase':>8} | {'Kernel (Orig)':>15} | {'Transformer':>15} | {'Mem K':>7} | {'Mem TF':>8}")
    print("-" * 115)

    for B in B_list:
        formulas = formulas_all[:B]
        try:
            phis = [from_string_to_formula(f) for f in formulas]
        except Exception as e:
            print(f"Errore parsing formule B={B}: {e}")
            continue

        for N in N_list:
            P = 1000
            k_res = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}
            tf_res = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}

            # --- 1. KERNEL TEST ---
            
            # --- PHASE A: KERNEL EMBEDDING ---
            try:
                torch.cuda.empty_cache()
                m_cpu = BaseMeasure(device="cpu")
                s1_cpu = m_cpu.sample(points=P, samples=N, varn=3)
                
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                s1_gpu = s1_cpu.to(DEVICE) 
                k1 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s1_gpu, samples=N, points=P, varn=3, integrate_time=True)
                
                sync()
                t0 = time.perf_counter()
                _ = k1._compute_robustness_time(phis) 
                sync()
                k_res["T_Emb"] = time.perf_counter() - t0
                k_res["M_Emb"] = get_gpu_memory() # Successo
                
                del s1_gpu, k1, s1_cpu
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Cattura il picco raggiunto prima del crash
                    k_res["M_Emb"] = torch.cuda.max_memory_allocated(device=DEVICE) / (1024 ** 2)
                    k_res["Status"] = "OOM_Emb"
                else:
                    k_res["Status"] = "Error_Emb"
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)

            # --- PHASE B: KERNEL SIMILARITY (Solo se l'embedding è riuscito) ---
            if k_res["Status"] == "Success":
                try:
                    torch.cuda.empty_cache()
                    m_cpu2 = BaseMeasure(device="cpu")
                    s2_cpu = m_cpu2.sample(points=P, samples=N, varn=3)
                    
                    torch.cuda.reset_peak_memory_stats(device=DEVICE)
                    s2_gpu = s2_cpu.to(DEVICE)
                    k2 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s2_gpu, samples=N, points=P, varn=3, integrate_time=True)
                    
                    sync()
                    t0 = time.perf_counter()
                    _ = k2.compute_bag_bag(phis, phis)
                    sync()
                    k_res["T_Sim"] = time.perf_counter() - t0
                    k_res["M_Sim"] = get_gpu_memory() # Successo
                    
                    del s2_gpu, k2, s2_cpu
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        k_res["M_Sim"] = torch.cuda.max_memory_allocated(device=DEVICE) / (1024 ** 2)
                        k_res["Status"] = "OOM_Sim"
                    else:
                        k_res["Status"] = "Error_Sim"
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device=DEVICE)
        
    #     for N in N_list:
    #         P = 1000
    #         k_res = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}
    #         tf_res = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}

    #         # --- 1. KERNEL TEST (SPOSTAMENTO DINAMICO VRAM) ---
    #         try:
    #             # Embedding
    #             torch.cuda.empty_cache()
    #             # Generazione in CPU per non inquinare le statistiche
    #             m_cpu = BaseMeasure(device="cpu")
    #             s1_cpu = m_cpu.sample(points=P, samples=N, varn=3)
                
    #             torch.cuda.reset_peak_memory_stats(device=DEVICE)
    #             s1_gpu = s1_cpu.to(DEVICE) # Carica ora
    #             k1 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s1_gpu, samples=N, points=P, varn=3, integrate_time=True)
                
    #             sync()
    #             t0 = time.perf_counter()
    #             _ = k1._compute_robustness_time(phis) 
    #             sync()
    #             k_res["T_Emb"], k_res["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()
                
    #             # Pulizia Aggressiva
    #             del s1_gpu, k1, s1_cpu
    #             torch.cuda.empty_cache()

    #             # Similarity
    #             m_cpu2 = BaseMeasure(device="cpu")
    #             s2_cpu = m_cpu2.sample(points=P, samples=N, varn=3)
                
    #             torch.cuda.reset_peak_memory_stats(device=DEVICE)
    #             s2_gpu = s2_cpu.to(DEVICE)
    #             k2 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s2_gpu, samples=N, points=P, varn=3, integrate_time=True)
                
    #             sync()
    #             t0 = time.perf_counter()
    #             _ = k2.compute_bag_bag(phis, phis)
    #             sync()
    #             k_res["T_Sim"], k_res["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()
                
    #             del s2_gpu, k2, s2_cpu
    #             torch.cuda.empty_cache()

    #         except RuntimeError as e:
    #             k_res["Status"] = "OOM" if "out of memory" in str(e).lower() else "Error"
    #             torch.cuda.empty_cache()
    #         except Exception as e:
    #             k_res["Status"] = "Error"

            # --- 2. TRANSFORMER TEST (ISOLAMENTO GARANTITO) ---
            try:
                # Embedding
                torch.cuda.empty_cache()
                sync()
                torch.cuda.reset_peak_memory_stats(device=DEVICE) 
                
                t0 = time.perf_counter()
                token_batches = tokenize_only(tokenizer, formulas, 128)
                _ = model_forward_only(model, token_batches)
                sync()
                # Ora M_Emb_TF misurerà SOLO il modello, dato che i segnali sono stati cancellati
                tf_res["T_Emb"], tf_res["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

                # Similarity
                torch.cuda.empty_cache()
                sync()
                torch.cuda.reset_peak_memory_stats(device=DEVICE) 
                
                t0 = time.perf_counter()
                token_batches_2 = tokenize_only(tokenizer, formulas, 128)
                z = model_forward_only(model, token_batches_2)
                z = F.normalize(z, p=2, dim=1)
                _ = z @ z.T 
                sync()
                tf_res["T_Sim"], tf_res["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

            except RuntimeError as e:
                tf_res["Status"] = "OOM" if "out of memory" in str(e).lower() else "Error"
                torch.cuda.empty_cache()
            except Exception as e:
                tf_res["Status"] = "Error"

            # Salvataggio e Logging
            row = {
                "B": B, "N": N, "K_Status": k_res["Status"], "TF_Status": tf_res["Status"],
                "T_Emb_K": k_res["T_Emb"], "M_Emb_K": k_res["M_Emb"],
                "T_Sim_K": k_res["T_Sim"], "M_Sim_K": k_res["M_Sim"],
                "T_Emb_TF": tf_res["T_Emb"], "M_Emb_TF": tf_res["M_Emb"],
                "T_Sim_TF": tf_res["T_Sim"], "M_Sim_TF": tf_res["M_Sim"]
            }
            results.append(row)
            pd.DataFrame(results).to_csv(output_file, index=False)

            # Print formatato
            k_t_emb = f"{k_res['T_Emb']:13.2f}s" if not np.isnan(k_res['T_Emb']) else k_res['Status']
            tf_t_emb = f"{tf_res['T_Emb']:13.2f}s" if not np.isnan(tf_res['T_Emb']) else tf_res['Status']
            print(f"{B:4} | {N:6} | EMBEDD  | {k_t_emb} | {tf_t_emb} | {k_res['M_Emb']:5.0f}MB | {tf_res['M_Emb']:6.0f}MB")
            
            k_t_sim = f"{k_res['T_Sim']:13.2f}s" if not np.isnan(k_res['T_Sim']) else k_res['Status']
            tf_t_sim = f"{tf_res['T_Sim']:13.2f}s" if not np.isnan(tf_res['T_Sim']) else tf_res['Status']
            print(f"{'':>4} | {'':>6} | SIMILAR | {k_t_sim} | {tf_t_sim} | {k_res['M_Sim']:5.0f}MB | {tf_res['M_Sim']:6.0f}MB")
            print("-" * 115)
            
    return pd.DataFrame(results)

# def run_benchmark_live_new(B_list, N_list):
#     results = []
#     output_file = os.path.join(RESULTS_DIR, "benchmark_final_cold_vram.csv")
    
#     print(f"🚀 Inizio Benchmark Scientifico (Isolamento Totale) su {DEVICE}...")
    
#     ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet")["train"]
#     from train_arch import BaseMeasure, from_string_to_formula
#     formulas_all = ds["formula"]

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
#     model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
#     model.eval()

#     print("\n" + "="*130)
#     print(f"{'B':>4} | {'N':>6} | {'Phase':>8} | {'Kernel (Orig)':>15} | {'Transformer':>15} | {'K RAM+VRAM':>12} | {'TF VRAM':>8}")
#     print("-" * 130)

#     for B in B_list:
#         formulas = formulas_all[:B]
#         try:
#             phis = [from_string_to_formula(f) for f in formulas]
#         except Exception as e:
#             print(f"Errore parsing formule B={B}: {e}")
#             continue

#         for N in N_list:
#             P = 1000
#             # Aggiungiamo campi per la RAM CPU nel dizionario risultati
#             k_res = {"T_Emb": np.nan, "M_VRAM_Emb": np.nan, "M_RAM_Emb": np.nan, 
#                      "T_Sim": np.nan, "M_VRAM_Sim": np.nan, "M_RAM_Sim": np.nan, "Status": "Success"}
#             tf_res = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}

#             # --- 1. KERNEL TEST ---
            
#             # --- PHASE A: KERNEL EMBEDDING ---
#             torch.cuda.empty_cache()
#             torch.cuda.reset_peak_memory_stats(device=DEVICE)
#             base_ram = get_cpu_ram_usage() 
#             try:
#                 m1 = BaseMeasure(device="cpu")
#                 s1_cpu = m1.sample(points=P, samples=N, varn=3)
#                 s1_gpu = s1_cpu.to(DEVICE) 
#                 k1 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s1_gpu, samples=N, points=P, varn=3, integrate_time=True)
                
#                 sync()
#                 t0 = time.perf_counter()
#                 _ = k1._compute_robustness_time(phis) 
#                 sync()
#                 k_res["T_Emb"] = time.perf_counter() - t0
#                 k_res["M_VRAM_Emb"] = get_gpu_memory()
#                 k_res["M_RAM_Emb"] = get_cpu_ram_usage() - base_ram # RAM usata dai tensori rhos
                
#                 del s1_gpu, k1, s1_cpu
#             except (RuntimeError, MemoryError) as e:
#                 # Se crasha, prendiamo l'ultimo picco registrato
#                 k_res["M_VRAM_Emb"] = torch.cuda.max_memory_allocated(device=DEVICE) / (1024 ** 2)
#                 k_res["M_RAM_Emb"] = get_cpu_ram_usage() - base_ram
#                 k_res["Status"] = "OOM_Emb"
#                 torch.cuda.empty_cache()

#             # --- PHASE B: KERNEL SIMILARITY ---
#             if k_res["Status"] == "Success":
#                 torch.cuda.empty_cache()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE)
#                 base_ram_sim = get_cpu_ram_usage()
#                 try:
#                     m2 = BaseMeasure(device="cpu")
#                     s2_cpu = m2.sample(points=P, samples=N, varn=3)
#                     s2_gpu = s2_cpu.to(DEVICE)
#                     k2 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s2_gpu, samples=N, points=P, varn=3, integrate_time=True)
                    
#                     sync()
#                     t0 = time.perf_counter()
#                     _ = k2.compute_bag_bag(phis, phis)
#                     sync()
#                     k_res["T_Sim"] = time.perf_counter() - t0
#                     k_res["M_VRAM_Sim"] = get_gpu_memory()
#                     k_res["M_RAM_Sim"] = get_cpu_ram_usage() - base_ram_sim
                    
#                     del s2_gpu, k2, s2_cpu
#                 except (RuntimeError, MemoryError) as e:
#                     k_res["M_VRAM_Sim"] = torch.cuda.max_memory_allocated(device=DEVICE) / (1024 ** 2)
#                     k_res["M_RAM_Sim"] = get_cpu_ram_usage() - base_ram_sim
#                     k_res["Status"] = "OOM_Sim"
#                     torch.cuda.empty_cache()

#             # --- 2. TRANSFORMER TEST ---
#             try:
#                 # Reset totale per misurare solo il modello
#                 torch.cuda.empty_cache()
#                 sync()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE) 
                
#                 t0 = time.perf_counter()
#                 token_batches = tokenize_only(tokenizer, formulas, 128)
#                 _ = model_forward_only(model, token_batches)
#                 sync()
#                 tf_res["T_Emb"], tf_res["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

#                 torch.cuda.empty_cache()
#                 sync()
#                 torch.cuda.reset_peak_memory_stats(device=DEVICE) 
                
#                 t0 = time.perf_counter()
#                 token_batches_2 = tokenize_only(tokenizer, formulas, 128)
#                 z = model_forward_only(model, token_batches_2)
#                 z = F.normalize(z, p=2, dim=1)
#                 _ = z @ z.T 
#                 sync()
#                 tf_res["T_Sim"], tf_res["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

#             except Exception as e:
#                 tf_res["Status"] = "Error"

#             # Salvataggio Dati
#             row = {
#                 "B": B, "N": N, "K_Status": k_res["Status"],
#                 "T_Emb_K": k_res["T_Emb"], "M_VRAM_K": k_res["M_VRAM_Emb"], "M_RAM_K": k_res["M_RAM_Emb"],
#                 "T_Sim_K": k_res["T_Sim"], "M_VRAM_Sim_K": k_res["M_VRAM_Sim"],
#                 "T_Emb_TF": tf_res["T_Emb"], "M_Emb_TF": tf_res["M_Emb"],
#                 "T_Sim_TF": tf_res["T_Sim"], "M_Sim_TF": tf_res["M_Sim"]
#             }
#             results.append(row)
#             pd.DataFrame(results).to_csv(output_file, index=False)

#             # Print formatato (Combiniamo RAM+VRAM per il Kernel nel log visivo)
#             k_mem_emb = f"{k_res['M_RAM_Emb']+k_res['M_VRAM_Emb']:5.0f}MB" if not np.isnan(k_res['M_RAM_Emb']) else "OOM"
#             print(f"{B:4} | {N:6} | EMBEDD  | {k_res['T_Emb']:13.2f}s | {tf_res['T_Emb']:13.2f}s | {k_mem_emb:>12} | {tf_res['M_Emb']:6.0f}MB")
            
#             k_mem_sim = f"{k_res['M_RAM_Sim']+k_res['M_VRAM_Sim']:5.0f}MB" if not np.isnan(k_res['M_RAM_Sim']) else "OOM"
#             print(f"{'':>4} | {'':>6} | SIMILAR | {k_res['T_Sim']:13.2f}s | {tf_res['T_Sim']:13.2f}s | {k_mem_sim:>12} | {tf_res['M_Sim']:6.0f}MB")
#             print("-" * 130)
            
#     return pd.DataFrame(results)


import gc

def run_benchmark_live_new(B_list, N_list):
    results = []
    output_file = os.path.join(RESULTS_DIR, "benchmark_final_cold_vram.csv")
    
    print(f"🚀 Inizio Benchmark Scientifico (Isolamento Totale) su {DEVICE}...")
    
    ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/saracandu/stl_updated/resolve/main/data/test-*.parquet")["train"]
    from train_arch import BaseMeasure, from_string_to_formula
    formulas_all = ds["formula"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model.eval()

    print("\n" + "="*145)
    print(f"{'B':>4} | {'N':>6} | {'Phase':>8} | {'Kernel Time':>15} | {'TF Time':>15} | {'K Net RAM':>12} | {'TF VRAM':>8}")
    print("-" * 145)

    for B in B_list:
        formulas = formulas_all[:B]
        try:
            phis = [from_string_to_formula(f) for f in formulas]
        except Exception as e:
            print(f"Errore parsing formule B={B}: {e}")
            continue

        for N in N_list:
            P = 1000
            k_res = {"T_Emb": np.nan, "M_VRAM_Emb": np.nan, "M_RAM_Emb": np.nan, 
                     "T_Sim": np.nan, "M_VRAM_Sim": np.nan, "M_RAM_Sim": np.nan, "Status": "Success"}
            tf_res = {"T_Emb": np.nan, "M_Emb": np.nan, "T_Sim": np.nan, "M_Sim": np.nan, "Status": "Success"}

            # --- 1. KERNEL TEST ---
            
            # --- PHASE A: KERNEL EMBEDDING ---
            gc.collect() 
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=DEVICE)
            base_ram_emb = get_cpu_ram_usage() 
            
            try:
                m1 = BaseMeasure(device="cpu")
                s1_cpu = m1.sample(points=P, samples=N, varn=3)
                s1_gpu = s1_cpu.to(DEVICE) 
                k1 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s1_gpu, samples=N, points=P, varn=3, integrate_time=True)
                
                sync()
                t0 = time.perf_counter()
                # Materializzazione rhos
                rhos, selfk, lengths = k1._compute_robustness_time(phis) 
                sync()
                
                k_res["T_Emb"] = time.perf_counter() - t0
                k_res["M_RAM_Emb"] = get_cpu_ram_usage() - base_ram_emb
                k_res["M_VRAM_Emb"] = get_gpu_memory() 
                
                # Pulizia per evitare accumulo tra A e B (ma vedi logica sotto per Similarity)
                del rhos, selfk, lengths, s1_gpu, k1, s1_cpu
                gc.collect()

            except (RuntimeError, MemoryError) as e:
                k_res["M_RAM_Emb"] = get_cpu_ram_usage() - base_ram_emb
                k_res["Status"] = "OOM_Emb"
                gc.collect()
                torch.cuda.empty_cache()

            # --- PHASE B: KERNEL SIMILARITY (Misura end-to-end per catturare il picco) ---
            if k_res["Status"] == "Success":
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=DEVICE)
                base_ram_sim = get_cpu_ram_usage()
                try:
                    m2 = BaseMeasure(device="cpu")
                    s2_cpu = m2.sample(points=P, samples=N, varn=3)
                    s2_gpu = s2_cpu.to(DEVICE)
                    k2 = StlKernel(measure=BaseMeasure(device=DEVICE), signals=s2_gpu, samples=N, points=P, varn=3, integrate_time=True)
                    
                    sync()
                    t0 = time.perf_counter()
                    # Qui compute_bag_bag ricalcola rhos internamente e poi fa la Gram Matrix.
                    # Mantenendo il riferimento a 'res', forziamo Python a non deallocare prima della misura.
                    res = k2.compute_bag_bag(phis, phis)
                    sync()
                    
                    k_res["T_Sim"] = time.perf_counter() - t0
                    # Catturiamo la RAM mentre 'res' (e potenzialmente i buffer di calcolo) è ancora attivo
                    k_res["M_RAM_Sim"] = get_cpu_ram_usage() - base_ram_sim
                    k_res["M_VRAM_Sim"] = get_gpu_memory()
                    
                    del s2_gpu, k2, s2_cpu, res
                    gc.collect()

                except (RuntimeError, MemoryError) as e:
                    k_res["M_RAM_Sim"] = get_cpu_ram_usage() - base_ram_sim
                    k_res["Status"] = "OOM_Sim"
                    gc.collect()

            # --- 2. TRANSFORMER TEST (Ricalcolo completo per Similarity) ---
            # --- PHASE A: EMBEDDING ONLY ---
            try:
                gc.collect()
                torch.cuda.empty_cache()
                sync()
                torch.cuda.reset_peak_memory_stats(device=DEVICE) 
                
                t0 = time.perf_counter()
                token_batches = tokenize_only(tokenizer, formulas, 128)
                _ = model_forward_only(model, token_batches)
                sync()
                tf_res["T_Emb"], tf_res["M_Emb"] = time.perf_counter() - t0, get_gpu_memory()

                # --- PHASE B: SIMILARITY (Encoding + Dot Product) ---
                gc.collect()
                torch.cuda.empty_cache()
                sync()
                torch.cuda.reset_peak_memory_stats(device=DEVICE) 
                
                t0 = time.perf_counter()
                token_batches_2 = tokenize_only(tokenizer, formulas, 128)
                z = model_forward_only(model, token_batches_2)
                z = F.normalize(z, p=2, dim=1)
                _ = z @ z.T 
                sync()
                
                tf_res["T_Sim"], tf_res["M_Sim"] = time.perf_counter() - t0, get_gpu_memory()

            except Exception as e:
                tf_res["Status"] = f"Error: {str(e)[:20]}"

            # Salvataggio Dati
            row = {
                "B": B, "N": N, "K_Status": k_res["Status"],
                "T_Emb_K": k_res["T_Emb"], "M_VRAM_Emb_K": k_res["M_VRAM_Emb"], "M_RAM_Emb_K": k_res["M_RAM_Emb"],
                "T_Sim_K": k_res["T_Sim"], "M_VRAM_Sim_K": k_res["M_VRAM_Sim"], "M_RAM_Sim_K": k_res["M_RAM_Sim"],
                "T_Emb_TF": tf_res["T_Emb"], "M_Emb_TF": tf_res["M_Emb"],
                "T_Sim_TF": tf_res["T_Sim"], "M_Sim_TF": tf_res["M_Sim"]
            }
            results.append(row)
            pd.DataFrame(results).to_csv(output_file, index=False)

            # Print formatato
            k_m_emb = f"{k_res['M_RAM_Emb']:5.0f}MB" if not np.isnan(k_res['M_RAM_Emb']) else "OOM"
            print(f"{B:4} | {N:6} | EMBEDD  | {k_res['T_Emb']:13.2f}s | {tf_res['T_Emb']:13.2f}s | {k_m_emb:>12} | {tf_res['M_Emb']:6.0f}MB")
            
            k_m_sim = f"{k_res['M_RAM_Sim']:5.0f}MB" if not np.isnan(k_res['M_RAM_Sim']) else "OOM"
            print(f"{'':>4} | {'':>6} | SIMILAR | {k_res['T_Sim']:13.2f}s | {tf_res['T_Sim']:13.2f}s | {k_m_sim:>12} | {tf_res['M_Sim']:6.0f}MB")
            print("-" * 145)
            
    return pd.DataFrame(results)

# if __name__ == "__main__":
#     B_LIST = [500, 1000, 2000, 4000]
#     N_LIST = [500, 1000, 2000, 4000, 8000, 16000]
    
#     df = run_benchmark_scientifico(B_LIST, N_LIST)
    
#     # Calcolo Speedup Finale Totale
#     df["T_Total_K"] = df["T_Emb_K"] + df["T_Sim_K"]
#     df["T_Total_TF"] = df["T_Emb_TF"] + df["T_Sim_TF"]
    
#     df.to_csv(os.path.join(RESULTS_DIR, "benchmark_final_decoupled_complete.csv"), index=False)

if __name__ == "__main__":
    B_LIST = [500, 1000, 2000, 4000]
    N_LIST = [500, 1000, 2000, 4000, 8000, 16000]
    
    df_results = run_benchmark_live_new(B_LIST, N_LIST)
    
#     # Calcolo Speedup Totale (Somma delle fasi / Somma delle fasi)
    df_results["T_Total_K"] = df_results["T_Emb_K"] + df_results["T_Sim_K"]
    df_results["T_Total_TF"] = df_results["T_Emb_TF"] + df_results["T_Sim_TF"]
    df_results["Speedup_Total"] = df_results["T_Total_K"] / df_results["T_Total_TF"]
                                 
    df_results.to_csv(os.path.join(RESULTS_DIR, "benchmark_final_cold_vram.csv"), index=False)