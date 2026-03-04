import os
import os
import wandb
from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModel, AutoTokenizer,
    Trainer, TrainingArguments
)
from datasets import load_dataset
from transformers import DefaultDataCollator

# --- CONFIG DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- IPERPARAMETRI ---
HPARAMS = {
    "encoder_model_id": "saracandu/stlenc-arch-cls",
    "decoder_model_id": "saracandu/stldec_arch",
    "batch_size": 128,
    "grad_acc": 4,
    "lr": 1e-4,
    "epochs": 5,
    "max_seq_len": 512,
    "project_name": "stl-decoder-training",
}

# ==========================================
# CUSTOM DATASET: Gestisce due tokenizer
# ==========================================
class STLDataset(Dataset):
    def __init__(self, formulas: List[str], enc_tokenizer, dec_tokenizer, max_len=512):
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.formulas = formulas
        self.max_len = max_len

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        formula = self.formulas[idx]
        
        # Tokenizzazione per l'Encoder (Input)
        enc_inputs = self.enc_tokenizer(
            formula,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Tokenizzazione per il Decoder (Labels/Output)
        dec_inputs = self.dec_tokenizer(
            formula,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc_inputs["input_ids"].squeeze(0),
            "attention_mask": enc_inputs["attention_mask"].squeeze(0),
            "labels": dec_inputs["input_ids"].squeeze(0) # Le labels usano il vocab del decoder
        }

# ==========================================
# CUSTOM TRAINER: Gestisce la Cross-Attention
# ==========================================
class STLDecTrainer(Trainer):
    def __init__(self, encoder, dec_tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder.to(DEVICE)
        self.encoder.eval() 
        self.dec_tokenizer = dec_tokenizer # Tokenizer del decoder per le labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # input_ids qui sono quelli dell'encoder (per generare il latente)
        enc_input_ids = inputs["input_ids"].to(DEVICE)
        enc_mask = inputs.get("attention_mask", None)
        # labels qui sono i token generati dal tokenizer del decoder
        labels = inputs["labels"].to(DEVICE)

        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=enc_input_ids, attention_mask=enc_mask)
            # Estrazione latente (CLS o Mean)
            latent = encoder_outputs.pooler_output if hasattr(encoder_outputs, "pooler_output") \
                     else encoder_outputs.last_hidden_state.mean(dim=1)
            latent = latent.unsqueeze(1) # [Batch, 1, Hidden]

        # Prepariamo le labels per il decoder (shift interno gestito da HF)
        # Sostituiamo il pad_token_id del decoder con -100 per ignorarlo nella loss
        dec_labels = labels.clone()
        dec_labels[dec_labels == self.dec_tokenizer.pad_token_id] = -100

        # Il decoder riceve le proprie labels come input_ids (Teacher Forcing)
        outputs = model(
            input_ids=labels, 
            encoder_hidden_states=latent.to(model.dtype),
            labels=dec_labels
        )
        
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

# ==========================================
# TRAINING FUNCTION
# ==========================================
def run_training():
    torch.cuda.empty_cache()

    wandb.init(
        project=HPARAMS["project_name"],
        config=HPARAMS,
        name=f"run-dual-tokenizer",
        reinit=True
    )

    # 1. Caricamento Tokenizer SEPARATI
    print("🔄 Caricamento tokenizer...")
    enc_tokenizer = AutoTokenizer.from_pretrained(HPARAMS["encoder_model_id"], trust_remote_code=True)
    dec_tokenizer = AutoTokenizer.from_pretrained(HPARAMS["decoder_model_id"], trust_remote_code=True)

    # 2. Caricamento Modelli
    encoder = AutoModel.from_pretrained(HPARAMS["encoder_model_id"], trust_remote_code=True)
    
    config = AutoConfig.from_pretrained(HPARAMS["decoder_model_id"], trust_remote_code=True)
    config.is_decoder = True
    config.add_cross_attention = True
    config.encoder_hidden_size = encoder.config.hidden_size
    config.pad_token_id = dec_tokenizer.pad_token_id
    config.vocab_size = len(dec_tokenizer) # Fondamentale: usa il vocab del decoder

    # Inizializziamo il decoder (config o pretrained revision)
    decoder = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    decoder.resize_token_embeddings(len(dec_tokenizer))
    decoder.to(DEVICE)

    # 3. Dataset con doppia tokenizzazione
    dataset = load_dataset("saracandu/stl_updated")
    train_dataset = STLDataset(
        dataset["train"]["formula"], 
        enc_tokenizer, 
        dec_tokenizer, 
        max_len=HPARAMS["max_seq_len"]
    )
    eval_dataset = STLDataset(
        dataset["test"]["formula"], 
        enc_tokenizer, 
        dec_tokenizer, 
        max_len=HPARAMS["max_seq_len"]
    )

    # 4. Training Arguments
    # training_args = TrainingArguments(
    #     output_dir="./results_dual_tokenizer",
    #     per_device_train_batch_size=HPARAMS["batch_size"],
    #     per_device_eval_batch_size=HPARAMS["batch_size"],
    #     gradient_accumulation_steps=HPARAMS["grad_acc"],
    #     num_train_epochs=HPARAMS["epochs"],
    #     learning_rate=HPARAMS["lr"],
    #     logging_steps=10, 
    #     eval_strategy="steps",
    #     eval_steps=100,
    #     bf16=True,
    #     push_to_hub=True,
    #     hub_model_id="saracandu/stldec_arch",
    #     report_to="wandb",
    #     gradient_checkpointing=True,
    #     remove_unused_columns=False, # Importante per passare le custom labels
    # )

    training_args = TrainingArguments(
    output_dir="./results_dual_tokenizer",

    per_device_train_batch_size=HPARAMS["batch_size"],
    per_device_eval_batch_size=HPARAMS["batch_size"],
    gradient_accumulation_steps=HPARAMS["grad_acc"],
    num_train_epochs=HPARAMS["epochs"],
    learning_rate=HPARAMS["lr"],

    logging_strategy="steps",
    logging_steps=50,                 # 🔥 log ogni step
    logging_first_step=True,
    logging_nan_inf_filter=False,

    eval_strategy="steps",
    eval_steps=100,                   # eval frequente

    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    save_safetensors=False,

    bf16=True,

    report_to="wandb",
    run_name="dual-tokenizer-full-logging",

    gradient_checkpointing=True,
    remove_unused_columns=False,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    )
    
    trainer = STLDecTrainer(
        encoder=encoder,
        dec_tokenizer=dec_tokenizer,
        model=decoder,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=dec_tokenizer, # Il tokenizer principale del trainer è quello del decoder
        data_collator=DefaultDataCollator()
    )

    print(f"🚀 Training iniziato su {DEVICE}...")
    trainer.train()
    
    wandb.finish()
    trainer.save_model("./stl_decoder_dual_tok")
    print("✅ Modello salvato!")

if __name__ == "__main__":
    run_training()