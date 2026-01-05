import copy
from collections import deque
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from train import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula

# ==========================================
# 2. PROCESSO DI FILTRAGGIO E UPDATE
# ==========================================

def update_repo_with_clean_data(repo_id):
    print(f"📥 Caricamento dataset da HF: {repo_id}...")
    ds_dict = load_dataset(repo_id)
    
    clean_splits = {}
    stats = {}

    for split in ds_dict.keys():
        print(f"🧹 Analisi e filtraggio dello split: {split}...")
        original_count = len(ds_dict[split])
        valid_rows = []
        
        for row in tqdm(ds_dict[split]):
            formula = row["formula_variant"]
            # Applichiamo il test di parsabilità
            if from_string_to_formula(formula):
                valid_rows.append(row)
        
        valid_count = len(valid_rows)
        discarded_count = original_count - valid_count
        discard_rate = (discarded_count / original_count) * 100
        
        stats[split] = {
            "original": original_count,
            "valid": valid_count,
            "discarded": discarded_count,
            "rate": discard_rate
        }
        
        clean_splits[split] = Dataset.from_list(valid_rows)

    # --- REPORT FINALE ---
    print("\n" + "="*40)
    print("📊 REPORT DI PULIZIA DATASET")
    print("="*40)
    for split, s in stats.items():
        print(f"Split [{split.upper()}]:")
        print(f"  - Totale originale: {s['original']}")
        print(f"  - Formule valide:   {s['valid']}")
        print(f"  - Formule scartate: {s['discarded']}")
        print(f"  - Percentuale scarto: {s['rate']:.2f}%")
        print("-" * 20)

    # Sovrascrittura della repo esistente
    new_ds = DatasetDict(clean_splits)
    print(f"\n🚀 Sovrascrittura della repo HF: {repo_id}...")
    # new_ds.push_to_hub(repo_id, max_shard_size="500MB")
    # print("✨ Dataset aggiornato con successo!")

if __name__ == "__main__":
    REPO = "saracandu/stl_formulae_variants"
    update_repo_with_clean_data(REPO)
