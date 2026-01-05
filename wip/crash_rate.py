import json
import copy
from datasets import load_dataset
from tqdm import tqdm
# Importiamo le classi e il parser dal tuo file train_new.py
from train_new import from_string_to_formula
# Importiamo le funzioni dal tuo data.py aggiornato
from data import simplify, generate_variants, to_string

def run_augmentation_on_dataset():
    print("🔄 Caricamento dataset da Hugging Face...")
    dataset = load_dataset("saracandu/stl_formulae", split="train")
    
    augmented_data = []
    errors = []
    success_count = 0
    crash_count = 0

    print(f"🚀 Inizio elaborazione di {len(dataset)} formule...")

    for i, entry in enumerate(tqdm(dataset)):
        # Il dataset saracandu/stl_formulae ha solitamente una colonna 'formula'
        raw_f = entry['formula']
        
        try:
            # 1. Tentativo di Parsing e Generazione Varianti
            # generate_variants include già il parsing e il simplify internamente
            varianti = generate_variants(raw_f, num_variants=3)
            
            if varianti:
                success_count += 1
                augmented_data.append({
                    "original": raw_f,
                    "variants": varianti
                })
            else:
                # Se la lista è vuota, generate_variants ha catturato un errore silente
                crash_count += 1
                errors.append({"index": i, "formula": raw_f, "error": "No variants generated"})
                
        except Exception as e:
            crash_count += 1
            errors.append({
                "index": i, 
                "formula": raw_f, 
                "error": str(e)
            })

    # --- REPORT FINALE ---
    print("\n" + "="*30)
    print("📊 REPORT FINALE")
    print("="*30)
    print(f"✅ Formule processate con successo: {success_count}")
    print(f"❌ Formule crashate: {crash_count}")
    if (success_count + crash_count) > 0:
        crash_rate = (crash_count / (success_count + crash_count)) * 100
        print(f"📉 Tasso di crash: {crash_rate:.2f}%")
    print("="*30)

    # Salvataggio risultati
    with open("augmented_results.json", "w") as f:
        json.dump(augmented_data, f, indent=2)
    
    with open("crash_report.json", "w") as f:
        json.dump(errors, f, indent=2)
        
    print("\n💾 Risultati salvati in 'augmented_results.json'")
    print("💾 Report errori salvato in 'crash_report.json'")

if __name__ == "__main__":
    run_augmentation_on_dataset()
