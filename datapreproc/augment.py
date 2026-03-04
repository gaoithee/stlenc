import os
import torch
import pandas as pd
import random
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from train_arch import from_string_to_formula
from datapreproc.perturb import generate_stratified_variants

# --- CONFIGURAZIONE ---
TOKENIZER_ID = "saracandu/stlenc"
SOURCE_DS = "saracandu/stl_new" 
DEST_DS = "saracandu/stl_high_complexity"
CHUNK_SIZE = 1000  # <--- Ridotto per vedere i file parquet comparire subito
SAVE_DIR = os.path.abspath("./stl_chunks_test_tmp")
MAX_TOKENS = 512
MAX_SIGNAL_POINTS = 1000 # <--- Aumentato drasticamente per validare formule profonde

# Inizializzazione
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
# Segnale più lungo per evitare IndexOutOfBounds durante il test di validità
DUMMY_SIGNAL = torch.randn(1, 1, MAX_SIGNAL_POINTS)

def is_valid_empirically(formula_str):
    if not formula_str: return False
    try:
        phi = from_string_to_formula(formula_str)
        if phi is None: return False
        with torch.no_grad():
            res = phi.quantitative(DUMMY_SIGNAL)
            # Verifica che il risultato esista e non sia una serie temporale svuotata
            return res is not None and (torch.is_tensor(res) and res.numel() > 0)
    except:
        return False

def run_augment_pipeline():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 Cartella creata: {SAVE_DIR}")

    print(f"📥 Caricamento sorgente: {SOURCE_DS}")
    ds_source = load_dataset(SOURCE_DS, split="test")
    
    col = "original" if "original" in ds_source.column_names else "formula"
    unique_formulas = ds_source.to_pandas()[col].unique().tolist()
    
    current_chunk_rows = []
    chunk_idx = 0
    temp_files = []
    total_valid_rows = 0

    pbar = tqdm(unique_formulas, desc="🚀 Generazione Traiettorie STL")
    
    for i, gold_f in enumerate(pbar):
        try:
            # Chiediamo 15 varianti
            variants = generate_stratified_variants(
                gold_f, 
                num_variants=15, 
                global_max_depth=20, 
                max_tokens=MAX_TOKENS
            )
            
            added_in_this_step = 0
            for v in variants:
                v_str = v["variant"]
                
                # Test empirico di calcolabilità
                if is_valid_empirically(v_str):
                    current_chunk_rows.append({
                        "formula": v_str,
                        "perturbation_type": v["type"],
                        "equivalent": float(v["label"]),
                        "original_formula": gold_f,
                        "depth": v.get("depth", 0)
                    })
                    added_in_this_step += 1
            
            total_valid_rows += added_in_this_step

            # Monitoraggio della resa media
            if (i + 1) % 10 == 0:
                avg_yield = total_valid_rows / (i + 1)
                pbar.set_postfix({
                    "yield": f"{avg_yield:.1f}/gold",
                    "buffer": len(current_chunk_rows),
                    "chunks": chunk_idx
                })

            # Salvataggio incrementale
            if len(current_chunk_rows) >= CHUNK_SIZE:
                path = os.path.join(SAVE_DIR, f"chunk_{chunk_idx}.parquet")
                pd.DataFrame(current_chunk_rows).to_parquet(path)
                temp_files.append(path)
                current_chunk_rows = []
                chunk_idx += 1

        except Exception:
            continue

    # Residuo finale
    if current_chunk_rows:
        path = os.path.join(SAVE_DIR, f"chunk_{chunk_idx}.parquet")
        pd.DataFrame(current_chunk_rows).to_parquet(path)
        temp_files.append(path)

    if not temp_files:
        print("❌ Nessun dato generato. Controlla la validità delle formule o la lunghezza del segnale.")
        return

    print(f"\n✅ Completato. Unione di {len(temp_files)} file...")
    final_ds = Dataset.from_parquet(temp_files)
    final_ds = final_ds.shuffle(seed=42)
    final_ds.push_to_hub(DEST_DS, split='test')

if __name__ == "__main__":
    run_augment_pipeline()