import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from train import from_string_to_formula
from data import generate_variants, to_string
from utils import simplify

def build_flat_dataset(source_repo):
    print(f"📥 Caricamento dataset: {source_repo}...")
    ds_dict = load_dataset(source_repo)
    output_dir = "stl_dataset_flat"
    os.makedirs(output_dir, exist_ok=True)

    for split_name in ds_dict.keys():
        print(f"🛠️ Elaborazione split: {split_name} (No Target Mode)...")
        processed_data = []
        
        for example in tqdm(ds_dict[split_name]):
            raw_formula = example.get('formula')
            embedding = example.get('embedding_1024')
            if not raw_formula: continue

            try:
                # 1. Recuperiamo le varie forme
                original_clean = " ".join(raw_formula.split())
                
                # Forma Super-Semplificata
                phi_obj = from_string_to_formula(raw_formula)
                phi_simple_obj = simplify(phi_obj)
                simplified = " ".join(to_string(phi_simple_obj).split())
                
                # 5 Varianti Complicate (num_variants=6 include il seme)
                variants = generate_variants(simplified, num_variants=6)
                
                # Creiamo il set unico di tutte le forme equivalenti
                all_forms = {original_clean, simplified}
                if variants:
                    for v in variants:
                        all_forms.add(" ".join(v.split()))
                
                # 2. Creiamo una riga nel dataset per OGNI forma equivalente
                # Tutte condivideranno lo stesso embedding e la stessa colonna 'original'
                for form in all_forms:
                    processed_data.append({
                        "formula_variant": form,
                        "original": original_clean,
                        "embedding_1024": embedding
                    })

            except Exception:
                continue
        
        # Salvataggio in Parquet
        df = pd.DataFrame(processed_data)
        output_path = os.path.join(output_dir, f"{split_name}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"✅ Split {split_name}: {len(df)} righe totali.")

    print(f"\n✨ Dataset flat completato! Ogni riga è una variante indipendente.")

if __name__ == "__main__":
    SOURCE = "saracandu/stl_formulae"
    try:
        build_flat_dataset(SOURCE)
    finally:
        os._exit(0)
