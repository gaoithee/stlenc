import os
import copy
import torch
import wandb
import torch.nn.functional as F
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

# --- CONFIGURAZIONE ---
REALNUM = Union[float, int]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARN = 3       
POINTS = 100
SAMPLES_FOR_KERNEL = 1000     
TEMPERATURE = 0.4    

# ==========================================
# 1. ARCHITETTURA NODI STL
# ==========================================

def eventually(x: Tensor, time_span: int) -> Tensor:
    return F.max_pool1d(x, kernel_size=time_span, stride=1)

class Node:
    def quantitative(self, x: Tensor, normalize: bool = False, evaluate_at_all_times: bool = False) -> Tensor:
        z: Tensor = self._quantitative(x, normalize)
        return z if evaluate_at_all_times else self._extract_semantics_at_time_zero(z)

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        return torch.reshape(x[:, 0, 0], (-1,))

class Atom(Node):
    def __init__(self, var_index: int, threshold: REALNUM, lte: bool = False) -> None:
        self.var_index, self.threshold, self.lte = var_index, threshold, lte
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        xj = x[:, self.var_index, :].view(x.size(0), 1, -1)
        z = -xj + self.threshold if self.lte else xj - self.threshold
        return torch.tanh(z) if normalize else z

class Not(Node):
    def __init__(self, child: Node) -> None: self.child = child
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: return -self.child._quantitative(x, normalize)

class And(Node):
    def __init__(self, left_child: Node, right_child: Node) -> None: self.left_child, self.right_child = left_child, right_child
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1, z2 = self.left_child._quantitative(x, normalize), self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        return torch.min(z1[:, :, :size], z2[:, :, :size])

class Or(Node):
    def __init__(self, left_child: Node, right_child: Node) -> None: self.left_child, self.right_child = left_child, right_child
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
        return self.right_child._quantitative(x[:, :, self.left_time_bound:], normalize)

# ==========================================
# 2. PARSER E KERNEL
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
    try:
        root_arity = 2 if st.startswith('(') else 1
        st_split = st.split()
        if root_arity <= 1:
            root_op_str = st_split[0]
            if root_op_str.startswith('x'):
                return Atom(var_index=int(st_split[0][2]), lte=(st_split[1] == '<='), threshold=float(st_split[2]))
            current_st = ' '.join(st_split[2:-1])
            if root_op_str == 'not': return Not(child=from_string_to_formula(current_st))
            un, run, l, r = set_time_thresholds(root_op_str)
            if root_op_str.startswith('eventually'): return Eventually(from_string_to_formula(current_st), un, run, l, r)
            return Globally(from_string_to_formula(current_st), un, run, l, r)
        else:
            current_st = st_split[1:-1]
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
                    op_str = current_st[children_range[0][1]+1] if var_child_idx == 1 else current_st[children_range[0][0]-1]
                    l_phi = from_string_to_formula(' '.join(current_st[:3] if var_child_idx == 0 else current_st[children_range[0][0]:children_range[0][1]+1]))
                    r_phi = from_string_to_formula(' '.join(current_st[-3:] if var_child_idx == 1 else current_st[children_range[0][0]:children_range[0][1]+1]))
                else:
                    op_str = current_st[children_range[0][1]+1]
                    l_phi = from_string_to_formula(' '.join(current_st[children_range[0][0]:children_range[0][1]+1]))
                    r_phi = from_string_to_formula(' '.join(current_st[children_range[1][0]:children_range[1][1]+1]))
            else:
                l_phi, r_phi, op_str = from_string_to_formula(' '.join(current_st[:3])), from_string_to_formula(' '.join(current_st[-3:])), current_st[3]
            if op_str == 'and': return And(l_phi, r_phi)
            if op_str == 'or': return Or(l_phi, r_phi)
            un, run, l, r = set_time_thresholds(op_str)
            return Until(l_phi, r_phi, un, run, l, r)
    except:
        return Atom(0, 0.0, True)

class BaseMeasure:
    def __init__(self, device="cpu"): self.device = device
    def sample(self, samples=1000, varn=3, points=100):
        signal = torch.rand(samples, varn, points, device=self.device)
        signal, _ = torch.sort(signal, 2)
        signal[:, :, 1:] = signal[:, :, 1:] - signal[:, :, :-1]
        signal[:, :, 0] = 0.0 + 1.0 * torch.randn(signal[:, :, 0].size(), device=self.device)
        derivs = torch.cumprod(2 * torch.bernoulli(0.9 * torch.ones(samples, varn, points, device=self.device)) - 1, 2)
        signal = torch.cumsum(signal * derivs * torch.pow(0.0 + 1.0 * torch.randn(samples, varn, 1, device=self.device), 2), 2)
        return signal

class StlKernel:
    def __init__(self, measure, sigma2=0.2, samples=1000, varn=3):
        self.sigma2, self.samples = sigma2, samples
        self.signals = measure.sample(samples=samples, varn=varn)
    def compute_bag(self, phis):
        rhos = torch.stack([p.quantitative(self.signals) for p in phis])
        selfk = torch.sum(rhos**2, dim=1, keepdim=True) / self.samples
        K = torch.matmul(rhos, rhos.T) / (self.samples * torch.sqrt(selfk * selfk.T) + 1e-8)
        return torch.exp(-(2.0 - 2 * K) / (2 * self.sigma2))

# ==========================================
# 3. TRAINER & DATA HANDLER
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
            K_target = self.stl_kernel.compute_bag(phi_objs)
            
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, "last_hidden_state") else outputs
        z = F.normalize(emb, p=2, dim=1)
        logits = torch.matmul(z, z.T) / TEMPERATURE  
        
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        
        mask = torch.eye(logits.shape[0], device=logits.device)
        numerator = exp_logits.diag()
        weighted_negatives = torch.sum(exp_logits * (1.0 - K_target) * (1.0 - mask), dim=1)
        loss = -torch.log(numerator / (numerator + weighted_negatives + 1e-8)).mean()
        
        if self.state.global_step % self.args.logging_steps == 0 and self.model.training:
            wandb.log({"train/loss": loss.item(), "train/avg_kernel_sim": K_target.mean().item()})
        return (loss, emb) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss, None, None)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Fix per TypeError: accetta logs e start_time opzionale."""
        if "loss" in logs and not self.model.training:
            logs["eval_loss"] = logs.pop("loss")
        super().log(logs, *args, **kwargs)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    wandb.init(project="stlenc-distillation", name="resume-epoch-5-to-10")
    
    model_id = "saracandu/stlenc-distilled"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    
    ds = load_dataset("saracandu/stl_formulae_variants")
    tokenized_ds = ds.map(lambda x: {
        **tokenizer(x["formula_variant"], truncation=True, max_length=512, padding="max_length"),
        "formula_str": x["formula_variant"]
    }, batched=True)
    
    tokenized_ds.set_format(type=None, columns=["input_ids", "attention_mask", "formula_str"])
    train_dataset = tokenized_ds["train"].shuffle(seed=42)
    eval_dataset = tokenized_ds["test"].shuffle(seed=42).select(range(3000))
    
    stl_kernel = StlKernel(measure=BaseMeasure(device=DEVICE), samples=SAMPLES_FOR_KERNEL, varn=VARN)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=128,
        num_train_epochs=10, 
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        report_to="wandb",
        push_to_hub=True,
        hub_model_id="saracandu/stlenc-distilled",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False
    )
    
    trainer = STLEncKernelTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, stl_kernel=stl_kernel,
        parse_fn=from_string_to_formula, tokenizer=tokenizer,
        data_collator=STLDataCollator(tokenizer=tokenizer)
    )
    
    checkpoint = True if os.path.exists("./results") and any("checkpoint" in d for d in os.listdir("./results")) else None
    print(f"🔄 Ripresa dal checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    wandb.finish()
