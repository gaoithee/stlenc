import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import ast
import numpy as np

# 1. Setup Modello Neurale
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "saracandu/stlenc-distilled-v2" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

# 2. Caricamento Dataset
print("Loading dataset...")
ds = load_dataset("saracandu/stl_formulae_variants", split="test")
sample_size = 100
test_sample = ds.shuffle(seed=42).select(range(sample_size))

# --- FUNZIONI DI SUPPORTO ---

def get_neural_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :] if hasattr(out, "last_hidden_state") else (out[:, 0, :] if out.dim() == 3 else out)
        return F.normalize(emb, p=2, dim=1)

def parse_given_embeddings(column_data):
    # Converte le stringhe "[0.1, 0.2, ...]" in tensori torch
    list_of_lists = [ast.literal_eval(x) if isinstance(x, str) else x for x in column_data]
    tensor_emb = torch.tensor(list_of_lists, dtype=torch.float32).to(device)
    return F.normalize(tensor_emb, p=2, dim=1)

def calculate_metrics(embs_v, embs_o):
    # Matrice di similarità (100x100)
    sim_matrix = torch.mm(embs_v, embs_o.t())
    # Diagonale: Variante vs Originale (Positivi)
    pos_sim = torch.diag(sim_matrix).mean().item()
    # Fuori diagonale: Variante vs Distrattori (Negativi)
    mask = torch.eye(sample_size, device=device).bool()
    neg_sim = sim_matrix[~mask].mean().item()
    return pos_sim, neg_sim

# --- ESECUZIONE ---

print("Processing Neural Embeddings...")
variants_neural = get_neural_embeddings(list(test_sample["formula_variant"]))
originals_neural = get_neural_embeddings(list(test_sample["original"]))

print("Processing Given Embeddings (embedding_1024)...")
# Assumendo che la colonna sia 'embedding_1024'
given_embs = parse_given_embeddings(list(test_sample["embedding_1024"]))
# Nota: se l'embedding nel dataset è associato solo all'originale o alla variante, 
# qui lo usiamo come riferimento fisso per la comparazione.
# Se hai embedding diversi per variante e originale nel dataset, dovresti caricarli entrambi.

# Calcolo metriche per il modello neurale
pos_n, neg_n = calculate_metrics(variants_neural, originals_neural)

# Calcolo metriche per l'embedding dato (usando se stesso come check di coerenza o vs varianti)
# Qui lo compariamo tra i vari campioni del batch per vedere la discriminatività "nativa"
sim_matrix_given = torch.mm(given_embs, given_embs.t())
mask = torch.eye(sample_size, device=device).bool()
pos_g = torch.diag(sim_matrix_given).mean().item() # Sarà 1.0 perché è lo stesso vettore
neg_g = sim_matrix_given[~mask].mean().item()

# --- RISULTATI ---
print("\n" + "="*45)
print(f"{'METRICA':<25} | {'NEURALE':<8} | {'DATO (1024)':<8}")
print("-" * 45)
print(f"{'Mean Pos (V vs O)':<25} | {pos_n:>8.4f} | {pos_g:>8.4f}")
print(f"{'Mean Neg (Distractors)':<25} | {neg_n:>8.4f} | {neg_g:>8.4f}")
print(f"{'Gap (Pos - Neg)':<25} | {pos_n-neg_n:>8.4f} | {pos_g-neg_g:>8.4f}")
print("="*45)
