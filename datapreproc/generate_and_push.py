import copy
import random
import re
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from train_temp3 import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula
from perturb import generate_stratified_variants

# --- CONFIGURAZIONE ---
TOKENIZER_ID = "saracandu/stlenc"
SOURCE_DS = "saracandu/stl_formulae_variants"
DEST_DS = "saracandu/stl_new"
CHUNK_SIZE = 1000
SAVE_DIR = "./stl_chunks_tmp"

# Caricamento Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

def process_formula(gold_str, tokenizer):
    """Genera 30 varianti con controlli su profondità e token."""
    # Passiamo esplicitamente tokenizer e parametri alla tua funzione in perturb.py
    variants = generate_stratified_variants(
        gold_str, 
        num_variants=30, 
        global_max_depth=20, 
        max_tokens=500
    )
    
    processed_data = []
    for v in variants:
        processed_data.append({
            "formula": v["variant"],
            "perturbation_type": v["type"],
            "equivalent": int(v["label"]),
            "original_formula": gold_str
        })
    return processed_data

def create_and_push_dataset():
    # 1. Preparazione ambiente
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # 2. Carica il dataset originale
    print("Caricamento dataset originale...")
    ds_source = load_dataset(SOURCE_DS, split="train")
    unique_formulas = ds_source.to_pandas()["original"].unique().tolist()
    print(f"Formule univoche trovate: {len(unique_formulas)}")

    current_chunk_rows = []
    chunk_idx = 0
    temp_files = []

    # 3. Generazione incrementale
    for i, gold_f in enumerate(tqdm(unique_formulas, desc="Generando perturbazioni")):
        try:
            new_rows = process_formula(gold_f, tokenizer)
            current_chunk_rows.extend(new_rows)

            # Salvataggio chunk su disco
            if (i + 1) % CHUNK_SIZE == 0:
                df_temp = pd.DataFrame(current_chunk_rows)
                path = os.path.join(SAVE_DIR, f"chunk_{chunk_idx}.parquet")
                df_temp.to_parquet(path)
                temp_files.append(path)
                
                # Svuota la RAM
                current_chunk_rows = []
                chunk_idx += 1
        except Exception as e:
            print(f"Errore alla formula {i}: {e}")
            continue

    # Ultimo residuo
    if current_chunk_rows:
        df_temp = pd.DataFrame(current_chunk_rows)
        path = os.path.join(SAVE_DIR, f"chunk_{chunk_idx}.parquet")
        df_temp.to_parquet(path)
        temp_files.append(path)

    # 4. Unione e Push (RAM-efficient)
    print(f"\nGenerazione completata. Unione di {len(temp_files)} file...")
    
    # Dataset.from_parquet mappa i file su disco senza caricarli tutti in RAM
    final_ds = Dataset.from_parquet(temp_files)
    
    # Applichiamo lo shuffle direttamente sul dataset HF (efficiente)
    print("Esecuzione shuffle del dataset...")
    final_ds = final_ds.shuffle(seed=42)
    
    # 5. Push su Hugging Face
    print(f"Pushing {len(final_ds)} righe su {DEST_DS}...")
    final_ds.push_to_hub(DEST_DS, split='train')
    
    # 6. Pulizia (opzionale)
    # import shutil
    # shutil.rmtree(SAVE_DIR)
    print("Completato!")

# ----------------------------------------------------------------------------------------------------

import os
import torch
from datasets import load_dataset
from train_temp3 import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula

# ==========================================
# CONFIGURAZIONE
# ==========================================
DATASET_REPO = "saracandu/stl_new"
MAX_SIGNAL_POINTS = 100 

# Creiamo un segnale dummy una sola volta per il test (costante per efficienza)
# Shape: [batch=1, channels=1, length=100]
DUMMY_SIGNAL = torch.randn(1, 1, MAX_SIGNAL_POINTS)

def is_valid_and_parsable(formula_str, max_points=100):
    """
    Verifica empirica: tenta di calcolare la robustezza.
    Se la libreria crasha (Pooling, Slicing, Indexing), restituisce False.
    """
    if not formula_str or not isinstance(formula_str, str):
        return False
    
    try:
        # 1. Tenta il parsing
        phi = from_string_to_formula(formula_str)
        if phi is None:
            return False
        
        # 2. Tenta il calcolo della robustezza (Simulazione reale)
        # Se la formula accede a indici fuori limite o riduce il segnale a zero,
        # qui scatterà un RuntimeError o IndexError.
        with torch.no_grad():
            # Il metodo quantitative() è quello che crasha durante il training
            res = phi.quantitative(DUMMY_SIGNAL)
            
            # 3. Verifica finale: se il risultato è vuoto o ha dimensioni errate
            if res is None or (torch.is_tensor(res) and res.numel() == 0):
                return False
                
        return True
        
    except Exception:
        # Se QUALSIASI cosa va storta (pooling -12, index out of bounds, etc.)
        # la formula viene scartata a monte.
        return False

# ==========================================
# PIPELINE DI SOVRASCRITTURA
# ==========================================

def overwrite_dataset():
    print(f"🚀 Download del dataset originale: {DATASET_REPO}...")
    # Carichiamo tutto il dataset (split train e test)
    ds = load_dataset(DATASET_REPO)

    # Identificazione colonna
    column_name = "formula" if "formula" in ds["train"].column_names else "formula_str"
    print(f"🔍 Colonna target: '{column_name}'")

    print(f"🛠 Inizio filtraggio empirico (Simulazione calcolo su {MAX_SIGNAL_POINTS} punti)...")
    
    # Usiamo num_proc=1 se vogliamo evitare problemi di memoria con torch in multiprocessing, 
    # ma con DUMMY_SIGNAL dovrebbe reggere bene anche in parallelo.
    filtered_ds = ds.filter(
        lambda x: is_valid_and_parsable(x[column_name], max_points=MAX_SIGNAL_POINTS),
        num_proc=os.cpu_count()
    )

    print("\n📊 Riepilogo Filtraggio:")
    for split in ds.keys():
        diff = len(ds[split]) - len(filtered_ds[split])
        print(f"   - {split}: {len(ds[split])} -> {len(filtered_ds[split])} (rimossi {diff})")

    print(f"\n📤 Sovrascrittura dataset su HF Hub: {DATASET_REPO}...")
    # Questo caricherà il dataset pulito sovrascrivendo quello vecchio
    filtered_ds.push_to_hub(DATASET_REPO, private=False)
    
    print("Spero per l'ultima volta")

if __name__ == "__main__":
    overwrite_dataset()


