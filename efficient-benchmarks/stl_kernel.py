import numpy as np
import torch


import os 
import copy 
import torch 
import numpy as np 
import wandb 
import tempfile 
import torch.nn.functional as F 
import functools
import multiprocessing 
from datetime import datetime 
from collections import deque 
from typing import List, Optional, Tuple, Union, Dict, Any 
from torch import Tensor 
from datasets import load_dataset 
from transformers import ( 
    AutoConfig,    
    AutoModel,     
    AutoTokenizer,     
    Trainer,     
    TrainingArguments,     
    DataCollatorWithPadding 
) 

# --- CONFIGURAZIONE DEVICE --- 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# --- OTTIMIZZAZIONE 1: CACHE DEL PARSER ---
@functools.lru_cache(maxsize=20000)
def cached_parse(formula_str: str):
    return from_string_to_formula(formula_str)

num_cores = max(1, multiprocessing.cpu_count() - 1) 

# --- CONFIGURAZIONE PATH E AMBIENTE --- 
BASE_DIR = "/share/ai-lab/scandussio/stlenc_arch_v2" 
HF_CACHE = os.path.join(BASE_DIR, "hf_cache") 
WANDB_DIR = os.path.join(BASE_DIR, "wandb") 
TMP_DIR = os.path.join(BASE_DIR, "tmp") 

for d in [HF_CACHE, WANDB_DIR, TMP_DIR]: 
    os.makedirs(d, exist_ok=True) 

os.environ["HF_HOME"] = HF_CACHE 
os.environ["HF_DATASETS_CACHE"] = HF_CACHE 
os.environ["WANDB_DIR"] = WANDB_DIR 
os.environ["TMPDIR"] = TMP_DIR 
tempfile.tempdir = TMP_DIR 

VARN = 3       
POINTS = 1000 
SAMPLES_FOR_KERNEL = 1000       
MODEL_ID = "saracandu/stlenc-arch-v2"  

# ========================================== 
# 1. LOGICA STL (OTTIMIZZATA GPU)
# ========================================== 
def eventually(x: Tensor, time_span: int) -> Tensor: 
    # MaxPool1d è estremamente efficiente su CUDA
    return F.max_pool1d(x, kernel_size=time_span, stride=1) 

class Node: 
    def quantitative(self, x: Tensor, normalize: bool = False, evaluate_at_all_times: bool = False) -> Tensor: 
        z: Tensor = self._quantitative(x, normalize) 
        return z if evaluate_at_all_times else self._extract_semantics_at_time_zero(z) 
    
    @staticmethod 
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor: 
        return torch.reshape(x[:, 0, 0], (-1,)) 

class Atom(Node): 
    def __init__(self, var_index: int, threshold: Union[float, int], lte: bool = False) -> None: 
        self.var_index, self.threshold, self.lte = var_index, threshold, lte 
        
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: 
        # x è già su DEVICE
        xj = x[:, self.var_index, :].view(x.size(0), 1, -1) 
        # Portiamo la soglia sullo stesso device del segnale per evitare sincronizzazioni CPU-GPU
        t = torch.tensor(self.threshold, device=x.device, dtype=x.dtype)
        z = -xj + t if self.lte else xj - t 
        return torch.tanh(z) if normalize else z 

class Not(Node): 
    def __init__(self, child: Node) -> None: self.child = child 
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: 
        return -self.child._quantitative(x, normalize) 

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
            # Implementazione Until richiede cautela su GPU per batch grandi
            return torch.cat([torch.max(torch.min(torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1), dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2) 
        return self.right_child._quantitative(x[:, :, self.left_time_bound:], normalize) 

# ========================================== 
# 2. PARSER E KERNEL (OTTIMIZZATO GPU)
# ========================================== 
def set_time_thresholds(st): 
    unbound, right_unbound = [True, False], [False, False] 
    l, r = 0, 0 
    if '[' in st and ']' in st: 
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
