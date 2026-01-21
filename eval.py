import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import random

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "saracandu/stlenc-distilled-v2" # Sostituisci con il tuo se caricato su HF
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

# 2. Caricamento Dataset
print("Loading dataset...")
ds = load_dataset("saracandu/stl_formulae_variants", split="test")

# Prendiamo un campione di test (es. 100 righe per non saturare la memoria)
sample_size = 100
test_sample = ds.shuffle(seed=42).select(range(sample_size))

def get_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        # Check per il tipo di output come abbiamo fatto prima
        emb = out.last_hidden_state[:, 0, :] if hasattr(out, "last_hidden_state") else (out[:, 0, :] if out.dim() == 3 else out)
        return F.normalize(emb, p=2, dim=1)


variants_list = [str(x) for x in list(test_sample["formula_variant"])]
originals_list = [str(x) for x in list(test_sample["original"])]

# 2. Ora chiama la funzione (assicurati che get_embeddings accetti text_list)
print("Computing embeddings...")
variants = get_embeddings(variants_list)
originals = get_embeddings(originals_list)

# 4. Calcolo Similarità
# Similarity Positiva: diagonale della matrice tra varianti e originali corrispondenti
pos_sims = torch.diag(torch.mm(variants, originals.t()))

# Similarity Negativa: confronto ogni variante con una formula random dal test set
# (shuffliamo gli indici degli originali assicurandoci che non coincidano)
indices = list(range(sample_size))
random.shuffle(indices)
neg_originals = originals[indices]
neg_sims = torch.diag(torch.mm(variants, neg_originals.t()))

# 5. Risultati
print("-" * 30)
print(f"Analisi su {sample_size} formule dal Test Set:")
print(f"Mean Positive Similarity (Variant vs Original): {pos_sims.mean().item():.4f}")
print(f"Mean Negative Similarity (Variant vs Random):   {neg_sims.mean().item():.4f}")
print("-" * 30)

# Esempio qualitativo del primo match
print(f"Example 1:")
print(f"V: {test_sample[0]['formula_variant']}")
print(f"O: {test_sample[0]['original']}")
print(f"Sim: {pos_sims[0].item():.4f}")
