import os
import copy
import pickle
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from collections import deque
from typing import List, Optional, Tuple, Union, Dict
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

# --- CONFIGURAZIONE ---
REALNUM = Union[float, int]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARN = 3  
POINTS = 100
SAMPLES_FOR_KERNEL = 1000 
TEMPERATURE = 0.07 # Corretto da 0.00 a 0.07 per stabilità numerica

# ==========================================
# 1. CODICE LOGICO STL (ORIGINALE + FIX DIMENSIONI)
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
    def __init__(self, var_index: int, threshold: REALNUM, lte: bool = False) -> None:
        super().__init__()
        self.var_index, self.threshold, self.lte = var_index, threshold, lte
    def _boolean(self, x: Tensor) -> Tensor:
        xj = x[:, self.var_index, :].view(x.size(0), 1, -1)
        return torch.le(xj, self.threshold) if self.lte else torch.ge(xj, self.threshold)
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        xj = x[:, self.var_index, :].view(x.size(0), 1, -1)
        z = -xj + self.threshold if self.lte else xj - self.threshold
        return torch.tanh(z) if normalize else z

class Not(Node):
    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child = child
    def _boolean(self, x: Tensor) -> Tensor: return ~self.child._boolean(x)
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: return -self.child._quantitative(x, normalize)

class And(Node):
    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child, self.right_child = left_child, right_child
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1, z2 = self.left_child._quantitative(x, normalize), self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        return torch.min(z1[:, :, :size], z2[:, :, :size])

class Or(Node):
    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child, self.right_child = left_child, right_child
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1, z2 = self.left_child._quantitative(x, normalize), self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        return torch.max(z1[:, :, :size], z2[:, :, :size])

class Globally(Node):
    def __init__(self, child, unbound=False, right_unbound=False, left_time_bound=0, right_time_bound=1, adapt_unbound=True):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.left_child, self.right_child, self.unbound, self.right_unbound = left, right, unbound, right_unbound
        self.left_time_bound, self.right_time_bound = left_time_bound, right_time_bound + 1
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1 = self.left_child._quantitative(x, normalize)
        z2 = self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        z1, z2 = z1[:, :, :size], z2[:, :, :size]
        if self.unbound:
            return torch.cat([torch.max(torch.min(torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1), dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        return self.right_child._quantitative(x[:, :, self.left_time_bound:], normalize)

# ==========================================
# 2. PARSER ORIGINALE
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

# ==========================================
# 3. MISURA E KERNEL
# ==========================================

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
        self.sigma2, self.samples = sigma2, samples
        self.signals = measure.sample(samples=samples, varn=varn)
    def compute_bag(self, phis, return_robustness=True):
        rhos = torch.stack([p.quantitative(self.signals) for p in phis])
        selfk = torch.sum(rhos**2, dim=1, keepdim=True) / self.samples
        K = torch.matmul(rhos, rhos.T) / (self.samples * torch.sqrt(selfk * selfk.T) + 1e-8)
        K_exp = torch.exp(-(2.0 - 2 * K) / (2 * self.sigma2))
        return (K_exp, rhos, selfk, None) if return_robustness else K_exp

# ==========================================
# 4. TRAINING COMPONENTS
# ==========================================

class STLDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        formula_strs = [f.pop("formula_str") for f in features]
        batch = super().__call__(features)
        batch["formula_str"] = formula_strs
        return batch

class STLEncKernelTrainer(Trainer):
    def __init__(self, stl_kernel, parse_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stl_kernel, self.parse_fn = stl_kernel, parse_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        formula_strs = inputs.pop("formula_str")
        if self.stl_kernel.signals.device != model.device:
            self.stl_kernel.signals = self.stl_kernel.signals.to(model.device)
        
        with torch.no_grad():
            phi_objs = [self.parse_fn(s) for s in formula_strs]
            K_target = self.stl_kernel.compute_bag(phi_objs, return_robustness=True)[0]
        
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, "last_hidden_state") else (outputs[:, 0, :] if outputs.dim() == 3 else outputs)
        z = F.normalize(emb, p=2, dim=1)
        
        # Similarità del modello
        model_sim = torch.matmul(z, z.T)
        logits = model_sim / TEMPERATURE
        
        exp_logits = torch.exp(logits)
        # Sostituiamo i negativi standard con quelli pesati dal kernel
        weighted_exp_sum = torch.sum(exp_logits * (1.0 - K_target), dim=1)
        loss = -torch.log(exp_logits.diag() / (weighted_exp_sum + 1e-8)).mean()
        
        # --- WANDB CUSTOM LOGGING ---
        if self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                # Calcoliamo quanto la similarità neurale è allineata a quella logica
                alignment_error = torch.abs(model_sim - K_target).mean()
                wandb.log({
                    "train/loss": loss.item(),
                    "train/kernel_alignment_error": alignment_error.item(),
                    "train/avg_kernel_similarity": K_target.mean().item(),
                    "train/avg_model_similarity": model_sim.mean().item()
                }, step=self.state.global_step)

        return (loss, emb) if return_outputs else loss

# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    # Inizializzazione WandB
    wandb.init(
        project="stlenc-distillation",
        name="kernel-weighted-infonce",
        config={
            "temperature": TEMPERATURE,
            "kernel_samples": SAMPLES_FOR_KERNEL,
            "varn": VARN,
            "learning_rate": 5e-5
        }
    )

    model_id, new_repo = "saracandu/stlenc", "saracandu/stlenc-kernel-distilled"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    
    ds = load_dataset("saracandu/stl_formulae", split="train")
    tokenized_ds = ds.map(lambda x: {**tokenizer(x["formula"], truncation=True, max_length=128, padding="max_length"), "formula_str": x["formula"]}, batched=True)
    tokenized_ds.set_format(type=None, columns=["input_ids", "attention_mask", "formula_str"])
    
    mu = BaseMeasure(device=DEVICE)
    stl_kernel = StlKernel(measure=mu, samples=SAMPLES_FOR_KERNEL, varn=VARN)
    
    

    trainer = STLEncKernelTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results", 
            per_device_train_batch_size=32, 
            num_train_epochs=10, 
            learning_rate=5e-5, 
            remove_unused_columns=False, 
            push_to_hub=True, 
            hub_model_id=new_repo, 
            report_to="wandb", # Abilita l'integrazione nativa
            logging_steps=10
        ),
        train_dataset=tokenized_ds, 
        stl_kernel=stl_kernel, 
        parse_fn=from_string_to_formula, 
        tokenizer=tokenizer, 
        data_collator=STLDataCollator(tokenizer=tokenizer)
    )
    
    trainer.train()
    wandb.finish()
