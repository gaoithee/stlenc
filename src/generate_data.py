import os
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from train import from_string_to_formula
from data import generate_variants, to_string
from utils import simplify

# Parametri di sistema per la validazione semantica
VARN = 3
POINTS = 100

def build_and_push_final(source_repo, target_repo):
    print(f"📥 Caricamento dataset sorgente: {source_repo}...")
    ds_dict = load_dataset(source_repo)
    
    final_dict = {}

    # Dummy signal per il test di "sostenibilità temporale" delle formule
    # (batch_size=1, num_vars=3, num_points=100)
    dummy_signal = torch.zeros(1, VARN, POINTS)

    for split_name in ds_dict.keys():
        print(f"🛠 Elaborazione split: {split_name}...")
        processed_data = []
        discarded_syntax = 0
        discarded_semantic = 0
        
        for example in tqdm(ds_dict[split_name]):
            raw_formula = example.get('formula')
            embedding = example.get('embedding_1024')
            if not raw_formula: continue

            try:
                # 1. Pulizia e parsing della forma originale
                original_clean = " ".join(raw_formula.split())
                phi_obj = from_string_to_formula(raw_formula)
                
                # 2. Generazione forma semplificata
                phi_simple_obj = simplify(phi_obj)
                simplified = " ".join(to_string(phi_simple_obj).split())
                
                # 3. Generazione varianti complicate (num_variants=6 include la base)
                variants = generate_variants(simplified, num_variants=6)
                
                # Creazione pool di candidati
                candidate_forms = {original_clean, simplified}
                if variants:
                    for v in variants:
                        candidate_forms.add(" ".join(v.split()))
                
                # --- VALIDAZIONE INTEGRATA ---
                for form in candidate_forms:
                    try:
                        # Test A: Il parser riesce a leggere la stringa?
                        test_phi = from_string_to_formula(form)
                        
                        # Test B: La formula "entra" nei 100 punti temporali?
                        # Questo previene il RuntimeError: max_pool1d() output size < 0
                        _ = test_phi.quantitative(dummy_signal)
                        
                        # Se entrambi i test passano, aggiungiamo la riga
                        processed_data.append({
                            "formula_variant": form,
                            "original": original_clean,
                            "embedding_1024": embedding
                        })
                        
                    except (IndexError, ValueError, RuntimeError):
                        # Se fallisce il parsing o il calcolo temporale, scartiamo la variante
                        if "IndexError" in str(Exception) or "ValueError" in str(Exception):
                            discarded_syntax += 1
                        else:
                            discarded_semantic += 1
                        continue

            except Exception:
                # Se fallisce il processamento della formula base di Irene, saltiamo l'intero gruppo
                continue
        
        # Creazione dello split filtrato
        final_dict[split_name] = Dataset.from_list(processed_data)
        print(f"✅ Split {split_name} completato.")
        print(f"   - Valide: {len(processed_data)}")
        print(f"   - Scartate (Sintassi): {discarded_syntax}")
        print(f"   - Scartate (Tempo/Runtime): {discarded_semantic}")

    # Push finale con sharding
    full_ds = DatasetDict(final_dict)
    print(f"🚀 Push su Hugging Face Hub: {target_repo}...")
    full_ds.push_to_hub(target_repo, max_shard_size="500MB")
    print("✨ Dataset aggiornato e pulito correttamente!")

if __name__ == "__main__":
    SOURCE = "saracandu/stl_formulae"
    TARGET = "saracandu/stl_formulae_variants"
    
    build_and_push_final(SOURCE, TARGET)
