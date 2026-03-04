import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from perturb import to_string

def merge_and_deduplicate_all_splits():
    repo_variants = "saracandu/stl_formulae_variants"
    repo_new = "saracandu/stl_new"
    target_repo = "saracandu/stl_new" # Sovrascriviamo per pulizia
    
    # Carichiamo entrambi i dataset completamente
    print("🚀 Caricamento dei dataset in corso...")
    ds_variants = load_dataset(repo_variants)
    ds_new = load_dataset(repo_new)
    
    merged_splits = {}
    
    # Iteriamo sugli split comuni (solitamente train e test)
    # Se uno split esiste solo in uno dei due, verrà comunque processato
    all_splits = set(ds_variants.keys()).union(set(ds_new.keys()))
    
    for split in all_splits:
        print(f"\n--- Elaborazione split: {split} ---")
        
        # 1. Recupero dei dati (se lo split non esiste in uno dei due, usiamo un DF vuoto)
        df_v = ds_variants[split].to_pandas() if split in ds_variants else pd.DataFrame()
        df_n = ds_new[split].to_pandas() if split in ds_new else pd.DataFrame()
        
        print(f"Righe originali -> Variants: {len(df_v)}, New: {len(df_n)}")
        
        # 2. Normalizzazione Nomi Colonne
        # Vogliamo che la colonna principale si chiami 'formula'
        for df in [df_v, df_n]:
            if not df.empty:
                if 'formula_variant' in df.columns:
                    df.rename(columns={'formula_variant': 'formula'}, inplace=True)
        
        # 3. Concatenazione
        df_combined = pd.concat([df_v, df_n], ignore_index=True)
        
        # 4. Rimozione Duplicati
        # Subset solo su 'formula' per garantire l'univocità semantica delle stringhe
        before_dedup = len(df_combined)
        df_combined.drop_duplicates(subset=['formula'], keep='first', inplace=True)
        after_dedup = len(df_combined)
        
        print(f"Rimossi {before_dedup - after_dedup} duplicati nello split {split}.")
        
        # Convertiamo di nuovo in Dataset HF
        merged_splits[split] = Dataset.from_pandas(df_combined)

    # 5. Creazione DatasetDict e Push
    print(f"\n📤 Push del dataset unificato su {target_repo}...")
    final_ds_dict = DatasetDict(merged_splits)
    final_ds_dict.push_to_hub(target_repo, private=False)
    
    print("✅ Operazione completata per tutti gli split!")

# if __name__ == "__main__":
#     merge_and_deduplicate_all_splits()
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

def consolidate_everything():
    repo_id = "saracandu/stl_new"
    
    print(f"🚀 Caricamento dataset da {repo_id}...")
    ds = load_dataset(repo_id, revision="a853f297a6c8577f22d81492df11ec14e6f5452e")
    
    final_splits = {}
    
    for split in ds.keys():
        print(f"--- Sistema split: {split} ---")
        df = ds[split].to_pandas()
        
        # 1. FUSIONE FORMULE ORIGINALI
        # Se 'original_formula' esiste, la usiamo come base e la riempiamo con 'original' dove mancano dati
        if 'original' in df.columns and 'original_formula' in df.columns:
            # combine_first: prende da original_formula, se NaN prende da original
            df['original_formula'] = df['original_formula'].combine_first(df['original'])
            # Ora che sono fuse, eliminiamo la colonna 'original'
            df.drop(columns=['original'], inplace=True)
            print("   ✅ Colonne 'original' e 'original_formula' fuse.")
        elif 'original' in df.columns and 'original_formula' not in df.columns:
            # Se esiste solo 'original', la rinominiamo per coerenza
            df.rename(columns={'original': 'original_formula'}, inplace=True)
            print("   ✅ Colonna 'original' rinominata in 'original_formula'.")

        # 2. SISTEMAZIONE LABEL BINARIE (equivalent)
        if 'equivalent' in df.columns:
            # Riempie i NaN con 1 (per il dataset nuovo che non le aveva)
            df['equivalent'] = df['equivalent'].fillna(1).astype(int)
            print("   ✅ Label 'equivalent' completate (NaN -> 1).")
        
        # 3. PULIZIA INDICI E COLONNE RESIDUE
        cols_to_drop = [c for c in df.columns if "Unnamed" in c or "index" in c]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        final_splits[split] = Dataset.from_pandas(df, preserve_index=False)

    # 4. PUSH FINALE
    print(f"\n📤 Push del dataset pulito e consolidato su {repo_id}...")
    DatasetDict(final_splits).push_to_hub(repo_id, private=False)
    print("✨ Operazione completata! Struttura finale: [formula, original_formula, equivalent]")

# ------------------------------------------------------------------------------------------------

import pandas as pd
import random
import copy
from datasets import load_dataset, Dataset, DatasetDict
from train_temp3 import from_string_to_formula, Atom, Globally, Eventually, Until

def apply_parametric_shift(node):
    node_cp = copy.deepcopy(node)
    def get_new_val(val, is_time=False):
        # 70% shift piccolo (1-20%), 30% shift grande (50-300%)
        scale = random.uniform(0.01, 0.20) if random.random() < 0.7 else random.uniform(0.5, 3.0)
        modifier = 1 + scale if random.random() > 0.5 else max(0.0, 1 - scale)
        new_val = float(val) * modifier
        return int(new_val) if is_time else "{:.4f}".format(new_val).replace("-0.0000", "0.0000")

    if isinstance(node_cp, Atom):
        node_cp.threshold = float(get_new_val(node_cp.threshold))
    elif isinstance(node_cp, (Globally, Eventually, Until)):
        node_cp.left_time_bound = get_new_val(node_cp.left_time_bound, True)
        if not node_cp.right_unbound:
            node_cp.right_time_bound = max(int(node_cp.left_time_bound) + 1, int(get_new_val(node_cp.right_time_bound, True)))
    
    # Ricorsione sui figli
    if hasattr(node_cp, 'child'): 
        node_cp.child = apply_parametric_shift(node_cp.child)
    elif hasattr(node_cp, 'left_child'):
        node_cp.left_child = apply_parametric_shift(node_cp.left_child)
        node_cp.right_child = apply_parametric_shift(node_cp.right_child)
    return node_cp

def augment_dataset():
    repo_id = "saracandu/stl_new"
    N_NEW_NEGATIVES = 500000
    
    print(f"🚀 Caricamento dataset...")
    ds = load_dataset(repo_id)
    df_train = ds['train'].to_pandas()
    
    # Selezioniamo solo le formule che sono equivalenti per perturbarle
    df_positives = df_train[df_train['equivalent'] == 1]
    
    print(f"🛠 Generazione di {N_NEW_NEGATIVES} perturbazioni parametriche...")
    new_rows = []
    
    # Campionamento con ripetizione se necessario, ma di solito abbiamo abbastanza basi
    samples = df_positives.sample(n=N_NEW_NEGATIVES, replace=True)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        try:
            phi = from_string_to_formula(row['formula'])
            phi_perturbed = apply_parametric_shift(phi)
            
            formula_str = to_string(phi_perturbed)
            
            new_rows.append({
                'formula': formula_str,
                'original_formula': row['formula'],
                'equivalent': 0 
            })
            
            if (i + 1) % 50000 == 0:
                print(f"   Progress: {i + 1}/{N_NEW_NEGATIVES}")
        except Exception as e:
            continue

    df_extra = pd.DataFrame(new_rows)
    
    # Unione e pulizia
    print("🧹 Unione e rimozione duplicati...")
    df_final_train = pd.concat([df_train, df_extra], ignore_index=True)
    df_final_train.drop_duplicates(subset=['formula'], keep='first', inplace=True)
    
    # Statistiche finali
    counts = df_final_train['equivalent'].value_counts()
    print(f"\n📊 Nuova distribuzione TRAIN:")
    print(f"   - Equivalenti (1): {counts.get(1, 0)}")
    print(f"   - Non Equivalenti (0): {counts.get(0, 0)}")
    
    # Salvataggio e Push
    print(f"\n📤 Push su Hugging Face...")
    ds_dict = DatasetDict({
        'train': Dataset.from_pandas(df_final_train, preserve_index=False),
        'test': ds['test'] # Il test lo lasciamo invariato
    })
    ds_dict.push_to_hub(repo_id)
    print("✅ Operazione completata!")

if __name__ == "__main__":
    augment_dataset()

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

def merge_with_explicit_labels():
    repo_variants = "saracandu/stl_formulae_variants"
    repo_new = "saracandu/stl_new"
    rev_new = "f179420bd62f32dc362e242ba6b0c50cf36aed02"
    
    print(f"🚀 Caricamento datasets...")
    ds_variants = load_dataset(repo_variants)
    ds_new = load_dataset(repo_new, revision=rev_new)
    
    final_splits = {}
    all_splits = set(ds_variants.keys()).union(set(ds_new.keys()))
    
    for split in all_splits:
        print(f"\n--- Split: {split} ---")
        
        # 1. Processiamo Variants (Tutte Label 1)
        df_v = pd.DataFrame()
        if split in ds_variants:
            df_v = ds_variants[split].to_pandas()
            df_v.rename(columns={'formula_variant': 'formula', 'original': 'original_formula'}, inplace=True)
            # Imponiamo label 1 a tutte le varianti
            df_v['equivalent'] = 1
            print(f"Variants: {len(df_v)} righe etichettate come 1 (Equivalent)")

        # 2. Processiamo New (Manteniamo label esistenti)
        df_n = pd.DataFrame()
        if split in ds_new:
            df_n = ds_new[split].to_pandas()
            # Se ci sono NaN in equivalent (formule nuove non labellate), assumiamo 1? 
            # O lasciamo come sono? Per sicurezza riempiamo i NaN con 1 se mancano.
            if 'equivalent' in df_n.columns:
                df_n['equivalent'] = df_n['equivalent'].fillna(1).astype(int)
            print(f"New: {len(df_n)} righe con label originali")

        # 3. Concatenazione
        df_combined = pd.concat([df_v, df_n], ignore_index=True)
        
        # 4. DEDUPLICAZIONE (Solo su 'formula')
        # Se una formula appare in entrambi, 'keep=first' terrà quella di Variants (Label 1)
        before = len(df_combined)
        df_combined.drop_duplicates(subset=['formula'], keep='first', inplace=True)
        after = len(df_combined)
        
        print(f"Rimossi {before - after} duplicati.")
        print(f"Distribuzione finale {split}:")
        print(df_combined['equivalent'].value_counts())

        # Pulizia finale colonne
        cols_to_keep = ['formula', 'original_formula', 'equivalent']
        df_combined = df_combined[[c for c in cols_to_keep if c in df_combined.columns]]

        final_splits[split] = Dataset.from_pandas(df_combined, preserve_index=False)

    # 5. Push
    print(f"\n📤 Push su {repo_new}...")
    ds_dict = DatasetDict(final_splits)
    ds_dict.push_to_hub(repo_new)
    print("✨ Merge e Labeling completato!")

# if __name__ == "__main__":
#     merge_with_explicit_labels()