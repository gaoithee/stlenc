import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
import copy
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import List, Union, Dict, Any, Optional, Tuple
from collections import deque
from Levenshtein import distance as levenshtein_dist
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARN = 3       
POINTS = 1000
SAMPLES_FOR_KERNEL = 1000    

MODEL_PATH = "saracandu/stlenc-arch"


# --- COPIA QUI LE TUE CLASSI E IL PARSER (Dal tuo file di training) ---
# ==========================================
# 1. LOGICA STL (Invariata)
# ==========================================
def eventually(x: Tensor, time_span: int) -> Tensor:
    return F.max_pool1d(x, kernel_size=time_span, stride=1)

class Node:
    def boolean(self, x: Tensor, evaluate_at_all_times: bool = False) -> Tensor:
        z: Tensor = self._boolean(x)
        return z if evaluate_at_all_times else self._extract_semantics_at_time_zero(z)
    def quantitative(self, x: Tensor, normalize: bool = False, evaluate_at_all_times: bool = False) -> Tensor:
        z: Tensor = self._quantitative(x, normalize)
        return z if evaluate_at_all_times else self._extract_semantics_at_time_zero(z)
    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        return torch.reshape(x[:, 0, 0], (-1,))

class Atom(Node):
    def __init__(self, var_index: int, threshold: Union[float, int], lte: bool = False) -> None:
        self.var_index, self.threshold, self.lte = var_index, threshold, lte
    def _boolean(self, x: Tensor) -> Tensor:
        xj = x[:, self.var_index, :].view(x.size(0), 1, -1)
        return torch.le(xj, self.threshold) if self.lte else torch.ge(xj, self.threshold)
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        xj = x[:, self.var_index, :].view(x.size(0), 1, -1)
        z = -xj + self.threshold if self.lte else xj - self.threshold
        return torch.tanh(z) if normalize else z

class Not(Node):
    def __init__(self, child: Node) -> None: self.child = child
    def _boolean(self, x: Tensor) -> Tensor: return ~self.child._boolean(x)
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: return -self.child._quantitative(x, normalize)

class And(Node):
    def __init__(self, left_child: Node, right_child: Node) -> None:
        self.left_child, self.right_child = left_child, right_child
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1, z2 = self.left_child._quantitative(x, normalize), self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        return torch.min(z1[:, :, :size], z2[:, :, :size])

class Or(Node):
    def __init__(self, left_child: Node, right_child: Node) -> None:
        self.left_child, self.right_child = left_child, right_child
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1, z2 = self.left_child._quantitative(x, normalize), self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        return torch.max(z1[:, :, :size], z2[:, :, :size])

class Globally(Node):
    def __init__(self, child, unbound=False, right_unbound=False, left_time_bound=0, right_time_bound=1, adapt_unbound=True):
        self.child, self.unbound, self.right_unbound = child, unbound, right_unbound
        self.left_time_bound, self.right_time_bound, self.adapt_unbound = left_time_bound, right_time_bound + 1, adapt_unbound
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1 = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            z = torch.cummin(torch.flip(z1, [2]), dim=2)[0] if self.adapt_unbound else torch.min(z1, 2, keepdim=True)[0]
            return torch.flip(z, [2]) if self.adapt_unbound else z
        return -eventually(-z1, self.right_time_bound - self.left_time_bound)

class Eventually(Node):
    def __init__(self, child, unbound=False, right_unbound=False, left_time_bound=0, right_time_bound=1, adapt_unbound=True):
        self.child, self.unbound, self.right_unbound = child, unbound, right_unbound
        self.left_time_bound, self.right_time_bound, self.adapt_unbound = left_time_bound, right_time_bound + 1, adapt_unbound
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1 = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            z = torch.cummax(torch.flip(z1, [2]), dim=2)[0] if self.adapt_unbound else torch.max(z1, 2, keepdim=True)[0]
            return torch.flip(z, [2]) if self.adapt_unbound else z
        return eventually(z1, self.right_time_bound - self.left_time_bound)

class Until(Node):
    def __init__(self, left, right, unbound=False, right_unbound=False, left_time_bound=0, right_time_bound=1):
        self.left_child, self.right_child, self.unbound, self.right_unbound = left, right, unbound, right_unbound
        self.left_time_bound, self.right_time_bound = left_time_bound, right_time_bound + 1
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1, z2 = self.left_child._quantitative(x, normalize), self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        z1, z2 = z1[:, :, :size], z2[:, :, :size]
        if self.unbound:
            return torch.cat([torch.max(torch.min(torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1), dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        return self.right_child._quantitative(x[:, :, self.left_time_bound:], normalize)

# ==========================================
# 2. PARSER E MISURA
# ==========================================
def set_time_thresholds(st):
    unbound, right_unbound = [True, False]
    l, r = [0, 0]
    if st[-1] == ']':
        unbound = False
        time_thresholds = st[st.index('[')+1:-1].split(",")
        l = int(time_thresholds[0])
        if time_thresholds[1] == 'inf': right_unbound = True
        else: r = int(time_thresholds[1]) - 1
    return unbound, right_unbound, l, r

def from_string_to_formula(st):
    root_arity = 2 if st.startswith('(') else 1
    st_split = st.split()
    if root_arity <= 1:
        root_op_str = copy.deepcopy(st_split[0])
        if root_op_str.startswith('x'):
            return Atom(var_index=int(st_split[0][2]), lte=(st_split[1] == '<='), threshold=float(st_split[2]))
        current_st = copy.deepcopy(st_split[2:-1])
        if root_op_str == 'not': return Not(child=from_string_to_formula(' '.join(current_st)))
        un, run, l, r = set_time_thresholds(root_op_str)
        if root_op_str.startswith('eventually'): return Eventually(from_string_to_formula(' '.join(current_st)), un, run, l, r)
        return Globally(from_string_to_formula(' '.join(current_st)), un, run, l, r)
    else:
        current_st = copy.deepcopy(st_split[1:-1])
        if '(' in current_st:
            par_queue, par_idx_list = deque(), []
            for i, sub in enumerate(current_st):
                if sub == '(': par_queue.append(i)
                elif sub == ')': par_idx_list.append(tuple([par_queue.pop(), i]))
            children_range = []
            for begin, end in sorted(par_idx_list):
                if children_range and children_range[-1][1] >= begin - 1: children_range[-1][1] = max(children_range[-1][1], end)
                else: children_range.append([begin, end])
            if len(children_range) == 1:
                var_child_idx = 1 if children_range[0][0] <= 1 else 0
                if children_range[0][0] != 0 and current_st[children_range[0][0]-1][0:2] in ['no', 'ev', 'al']: children_range[0][0] -= 1
                l_str = current_st[:3] if var_child_idx == 0 else current_st[children_range[0][0]:children_range[0][1]+1]
                r_str = current_st[-3:] if var_child_idx == 1 else current_st[children_range[0][0]:children_range[0][1]+1]
                op_str = current_st[children_range[0][1]+1] if var_child_idx == 1 else current_st[children_range[0][0]-1]
            else:
                if children_range[0][0] != 0 and current_st[children_range[0][0]-1][0:2] in ['no', 'ev', 'al']: children_range[0][0] -= 1
                if current_st[children_range[1][0]-1][0:2] in ['no', 'ev', 'al']: children_range[1][0] -= 1
                op_str, l_str, r_str = current_st[children_range[0][1]+1], current_st[children_range[0][0]:children_range[0][1]+1], current_st[children_range[1][0]:children_range[1][1]+1]
        else: l_str, r_str, op_str = current_st[:3], current_st[-3:], current_st[3]
        l_phi, r_phi = from_string_to_formula(' '.join(l_str)), from_string_to_formula(' '.join(r_str))
        if op_str == 'and': return And(l_phi, r_phi)
        if op_str == 'or': return Or(l_phi, r_phi)
        un, run, l, r = set_time_thresholds(op_str)
        return Until(l_phi, r_phi, un, run, l, r)

class BaseMeasure:
    def __init__(self, mu0=0.0, sigma0=1.0, mu1=0.0, sigma1=1.0, q=0.1, q0=0.5, device="cpu"):
        self.mu0, self.sigma0, self.mu1, self.sigma1, self.q, self.q0, self.device = mu0, sigma0, mu1, sigma1, q, q0, device
    def sample(self, samples=1000, varn=3, points=100):
        signal = torch.rand(samples, varn, points, device=self.device)
        signal[:, :, 0], signal[:, :, -1] = 0.0, 1.0
        signal, _ = torch.sort(signal, 2)
        signal[:, :, 1:] = signal[:, :, 1:] - signal[:, :, :-1]
        signal[:, :, 0] = self.mu0 + self.sigma0 * torch.randn(signal[:, :, 0].size(), device=self.device)
        derivs = torch.cumprod(2 * torch.bernoulli((1-self.q)*torch.ones(samples, varn, points, device=self.device)) - 1, 2)
        derivs[:, :, 0] = 2 * torch.bernoulli(torch.full((samples, varn), self.q0, device=self.device)) - 1
        signal = torch.cumsum(signal * derivs * torch.pow(self.mu1 + self.sigma1 * torch.randn(samples, varn, 1, device=self.device), 2), 2)
        return signal

class StlKernel:
    def __init__(self, measure, sigma2=0.2, samples=1000, varn=3):
        self.measure = measure
        self.sigma2, self.samples, self.varn = sigma2, samples, varn
        self.signals = measure.sample(samples=samples, varn=varn, points=POINTS)
    def compute_bag(self, phis, return_robustness=True):
        rhos = torch.stack([p.quantitative(self.signals) for p in phis])
        selfk = torch.sum(rhos**2, dim=1, keepdim=True) / self.samples
        K = torch.matmul(rhos, rhos.T) / (self.samples * torch.sqrt(selfk * selfk.T) + 1e-8)
        K_exp = torch.exp(-(2.0 - 2 * K) / (2 * self.sigma2))
        return (K_exp, rhos, selfk, None) if return_robustness else K_exp


# --- CONFIGURAZIONE DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = "/share/ai-lab/scandussio/stlenc_arch_mean" 
MODEL_PATH = "saracandu/stlenc-arch-v2"

# ==========================================
# UTILITY DI EMBEDDING
# ==========================================
# def get_embeddings(model, tokenizer, formulas: List[str]):
#     model.eval()
#     inputs = tokenizer(formulas, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Estraiamo l'embedding
#         emb = outputs.pooler_output
        
#         # Forza la forma (BatchSize, EmbeddingDim)
#         if emb.ndim == 1:
#             emb = emb.unsqueeze(0)
            
#         # Normalizzazione L2 sulla sfera unitaria
#         emb = F.normalize(emb, p=2, dim=1)
#     return emb

def get_embeddings(model, tokenizer, formulas: List[str]):
    model.eval()
    inputs = tokenizer(formulas, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Se la versione vecchia restituisce direttamente il Tensor:
        if isinstance(outputs, torch.Tensor):
            emb = outputs
        else:
            # Fallback se restituisce un dizionario o oggetto
            emb = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            
        # Normalizzazione L2 (essenziale per la Cosine Similarity)
        emb = F.normalize(emb, p=2, dim=1)
    return emb

KERNEL_RUNS = 3

from Levenshtein import distance as levenshtein_dist
import random
from tqdm import tqdm
import numpy as np
import torch

from Levenshtein import distance as levenshtein_dist
import random
from tqdm import tqdm
import numpy as np
import torch

from Levenshtein import distance as levenshtein_dist
import random
from tqdm import tqdm
import numpy as np
import torch
from Levenshtein import distance as levenshtein_dist
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from Levenshtein import distance as levenshtein_dist
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def evaluate_semantic_classes(model_name, n_samples=100, edit_top_k=10, kernel_thresh=0.75):
    print("\n" + "="*80)
    print(f"EVALUATING MODEL: {model_name}")
    print("="*80)

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="66561dfa0430a21e6c7f45fc36b937fa077f2003")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision="66561dfa0430a21e6c7f45fc36b937fa077f2003").to(DEVICE)
    model.eval()

    # --- Load dataset ---
    dataset = load_dataset("saracandu/stl_updated", split="test")
    df = dataset
    df_logic = [row for row in df if row['perturbation_type'] == 'logic_exact']
    df_non_logic = [row for row in df if row['perturbation_type'] != 'logic_exact']

    # --- Initialize kernel ---
    stl_kernel = StlKernel(BaseMeasure(device=DEVICE), samples=1000, varn=3)

    # --- Containers for metrics ---
    results = {k: [] for k in [
        "sims_logic", "latent_dists_logic", "rel_latent_logic",
        "kernels_logic", "kernel_dists_logic", "rel_kernel_logic",
        "sims_random", "latent_dists_random", "rel_latent_random",
        "kernels_random", "kernel_dists_random", "rel_kernel_random",
        "sims_edit", "latent_dists_edit", "rel_latent_edit",
        "kernels_edit", "kernel_dists_edit", "rel_kernel_edit"
    ]}

    # --- Helper ---
    def latent_dist(e1, e2):
        return torch.norm(e1 - e2, p=2).item()

    # --- 1) Semantic equivalence ---
    for row in tqdm(df_logic, desc="Processing logic_exact"):
        try:
            f1 = " ".join(row['original_formula'].replace('(', ' ( ').replace(')', ' ) ').split())
            f2 = " ".join(row['formula'].replace('(', ' ( ').replace(')', ' ) ').split())
            phi1, phi2 = from_string_to_formula(f1), from_string_to_formula(f2)
            e1, e2 = get_embeddings(model, tokenizer, [f1]), get_embeddings(model, tokenizer, [f2])

            sim = torch.matmul(e1, e2.t()).item()
            dist_lat = latent_dist(e1, e2)
            k_val = stl_kernel.compute_bag([phi1, phi2], return_robustness=True)[0][0,1].item()

            results["sims_logic"].append(sim)
            results["latent_dists_logic"].append(dist_lat)
            results["kernels_logic"].append(k_val)
            results["kernel_dists_logic"].append(1 - k_val)
        except Exception:
            continue

    # Relative distances
    ld = np.array(results["latent_dists_logic"])
    kd = np.array(results["kernel_dists_logic"])
    results["rel_latent_logic"] = (ld / ld.max()).tolist() if len(ld) > 0 else []
    results["rel_kernel_logic"] = (kd / kd.max()).tolist() if len(kd) > 0 else []

    # --- 2) Random non-equivalent ---
    indices = random.sample(range(len(df_non_logic)), min(n_samples, len(df_non_logic)))
    for idx in tqdm(indices, desc="Processing random non-equivalent"):
        try:
            f_orig = " ".join(df_non_logic[idx]['formula'].replace('(', ' ( ').replace(')', ' ) ').split())
            phi_orig = from_string_to_formula(f_orig)
            e1 = get_embeddings(model, tokenizer, [f_orig])

            # Random formula
            f_rand_row = random.choice(df_non_logic)
            f_rand = " ".join(f_rand_row['formula'].replace('(', ' ( ').replace(')', ' ) ').split())
            phi_rand = from_string_to_formula(f_rand)
            e2 = get_embeddings(model, tokenizer, [f_rand])

            sim = torch.matmul(e1, e2.t()).item()
            dist_lat = latent_dist(e1, e2)
            k_val = stl_kernel.compute_bag([phi_orig, phi_rand], return_robustness=True)[0][0,1].item()

            results["sims_random"].append(sim)
            results["latent_dists_random"].append(dist_lat)
            results["kernels_random"].append(k_val)
            results["kernel_dists_random"].append(1 - k_val)
        except Exception:
            continue

    # Relative distances
    ld = np.array(results["latent_dists_random"])
    kd = np.array(results["kernel_dists_random"])
    results["rel_latent_random"] = (ld / ld.max()).tolist() if len(ld) > 0 else []
    results["rel_kernel_random"] = (kd / kd.max()).tolist() if len(kd) > 0 else []

    # --- 3) Edit-distance non-equivalent ---
    for idx in tqdm(indices, desc="Processing edit-distance non-equivalent"):
        try:
            f_orig = " ".join(df_non_logic[idx]['formula'].replace('(', ' ( ').replace(')', ' ) ').split())
            phi_orig = from_string_to_formula(f_orig)
            e1 = get_embeddings(model, tokenizer, [f_orig])

            # Closest syntactic candidates
            candidates = []
            min_edit = float('inf')
            for row in df_non_logic:
                f_cand = row['formula']
                d = levenshtein_dist(f_orig, f_cand)
                if 0 < d < min_edit:
                    min_edit = d
                    candidates = [f_cand]
                elif d == min_edit:
                    candidates.append(f_cand)

            # Semantic filtering
            selected = None
            for f_cand in candidates[:edit_top_k]:
                try:
                    phi_cand = from_string_to_formula(" ".join(f_cand.replace('(', ' ( ').replace(')', ' ) ').split()))
                    k_val = stl_kernel.compute_bag([phi_orig, phi_cand], return_robustness=True)[0][0,1].item()
                    if k_val < kernel_thresh:
                        selected = f_cand
                        phi_cand_selected = phi_cand
                        break
                except Exception:
                    continue

            if selected is not None:
                e2 = get_embeddings(model, tokenizer, [selected])
                sim = torch.matmul(e1, e2.t()).item()
                dist_lat = latent_dist(e1, e2)
                k_val = stl_kernel.compute_bag([phi_orig, phi_cand_selected], return_robustness=True)[0][0,1].item()

                results["sims_edit"].append(sim)
                results["latent_dists_edit"].append(dist_lat)
                results["kernels_edit"].append(k_val)
                results["kernel_dists_edit"].append(1 - k_val)
        except Exception:
            continue

    # Relative distances
    ld = np.array(results["latent_dists_edit"])
    kd = np.array(results["kernel_dists_edit"])
    results["rel_latent_edit"] = (ld / ld.max()).tolist() if len(ld) > 0 else []
    results["rel_kernel_edit"] = (kd / kd.max()).tolist() if len(kd) > 0 else []

    return results

# -----------------------------------------------------
# RUN BOTH MODELS
# -----------------------------------------------------
# models = [
#     "saracandu/stlenc-arch-v2",
#     "saracandu/stlenc-arch-cls"
# ]

models = ["saracandu/stlenc-arch"]

# -----------------------------------------------------
# PRINT AGGREGATE METRICS
# -----------------------------------------------------
for m in models:
    print("\n" + "="*80)
    print(f"RESULTS FOR MODEL: {m}")
    print("="*80)
    res = evaluate_semantic_classes(m)

    for cls, name in zip(["logic", "random", "edit"], 
                         ["Semantic Equivalence", "Random Non-Equivalent", "Edit-Distance Non-Equivalent"]):
        sims = np.array(res[f"sims_{cls}"])
        latent = np.array(res[f"latent_dists_{cls}"])
        rel_latent = np.array(res[f"rel_latent_{cls}"])
        kernels = np.array(res[f"kernels_{cls}"])
        kernel_dists = np.array(res[f"kernel_dists_{cls}"])
        rel_kernel = np.array(res[f"rel_kernel_{cls}"])

        print(f"\n--- {name} Pairs ---")
        print("Samples:", len(sims))
        print("Mean Neural Similarity:", sims.mean() if len(sims)>0 else 0)
        print("Mean Latent Distance:", latent.mean() if len(latent)>0 else 0)
        print("Mean Relative Latent Distance:", rel_latent.mean() if len(rel_latent)>0 else 0)
        print("Mean Kernel Similarity:", kernels.mean() if len(kernels)>0 else 0)
        print("Mean Kernel Distance:", kernel_dists.mean() if len(kernel_dists)>0 else 0)
        print("Mean Relative Kernel Distance:", rel_kernel.mean() if len(rel_kernel)>0 else 0)
        print("MAE (semantic gap):", np.mean(np.abs(sims - kernels)) if len(sims)>0 else 0)
        print("Invariance Rate (>0.85):", np.mean(sims > 0.85) if len(sims)>0 else 0)