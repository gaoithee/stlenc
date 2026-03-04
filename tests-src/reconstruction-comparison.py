import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# --- CONFIGURAZIONE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENC_ID = "saracandu/stlenc-arch-cls"
DEC_ID = "saracandu/stldec_arch"
DATASET_ID = "saracandu/stl_formulae"
BATCH_SIZE = 10  # Aumentato per velocità, riduci se vai in Out of Memory
MAX_LENGTH = 512

def process_test_set():
    print(f"🌍 Caricamento dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="test")
    test_formulas = dataset["formula"] # Assumendo che la colonna si chiami 'formula'
    
    print(f"🚀 Caricamento modelli su {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(ENC_ID, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(ENC_ID, trust_remote_code=True).to(DEVICE)
    # revision="a885fc854fb67d321639eae41d3e73837268cab7"
    config = AutoConfig.from_pretrained(DEC_ID, trust_remote_code=True)
    config.is_decoder = True
    config.add_cross_attention = True
    config.use_cache = False 
    
    decoder = AutoModelForCausalLM.from_pretrained(DEC_ID, config=config, trust_remote_code=True).to(DEVICE)
    
    encoder.eval()
    decoder.eval()

    # Prepariamo il DataLoader (usiamo una semplice lista di stringhe)
    dataloader = DataLoader(test_formulas, batch_size=BATCH_SIZE, shuffle=False)

    all_originals = []
    all_reconstructed = []

    print(f"🧪 Inizio inferenza su {len(test_formulas)} formule...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Decoding Batches"):
            # 1. Encoding
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            encoder_outputs = encoder(**inputs)
            
            # 2. Latent Projection (Pooling)
            if hasattr(encoder_outputs, "pooler_output") and encoder_outputs.pooler_output is not None:
                latent = encoder_outputs.pooler_output
            else:
                latent = encoder_outputs.last_hidden_state.mean(dim=1)
            
            latent = F.normalize(latent, p=2, dim=1).unsqueeze(1) 

            # 3. Generazione Auto-regressiva
            start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else (tokenizer.pad_token_id or 0)
            input_ids = torch.full((len(batch), 1), start_token, device=DEVICE)

            output_ids = decoder.generate(
                input_ids=input_ids,
                encoder_hidden_states=latent.to(decoder.dtype),
                max_length=MAX_LENGTH,
                do_sample=False, # Greedy search per massima coerenza
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # 4. Decoding e Post-processing
            decoded_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # --- CORREZIONE QUI ---
            batch_reconstructed = []
            for raw_gen in decoded_batch:
                # Trasforma "@ @ x_ 0" in "x_0" ecc.
                # Rimuoviamo gli spazi originali e sostituiamo @ con spazio, poi puliamo i doppi spazi
                clean = raw_gen.replace(" ", "").replace("@", " ").strip()
                batch_reconstructed.append(clean)

            # DEBUG: mostra i primi esempi correttamente
            if len(all_originals) == 0:
                print("\n🔍 Esempi di ricostruzione:")
                for i in range(min(5, len(batch))):
                    print("-" * 40)
                    print("ORIG:", batch[i])
                    print("GENc :", batch_reconstructed[i]) # Accediamo alla lista corretta
                print("-" * 40)
            
            # Aggiungi i risultati del batch alle liste globali
            all_originals.extend(batch)
            all_reconstructed.extend(batch_reconstructed)

    # --- SALVATAGGIO ---
    results_df = pd.DataFrame({
        "formula_originale": all_originals,
        "formula_ricostruita": all_reconstructed
    })
    
    # Calcolo una metrica veloce di "Exact Match"
    exact_matches = (results_df["formula_originale"] == results_df["formula_ricostruita"]).sum()
    accuracy = (exact_matches / len(results_df)) * 100

    output_file = "stl_reconstruction_results_new.csv"
    results_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    process_test_set()
