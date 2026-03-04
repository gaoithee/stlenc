import torch
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda"

class FormulaDataset(Dataset):
    def __init__(self, formulas, tokenizer, max_length=512):
        self.formulas = formulas
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        return self.formulas[idx]

def tokenize_only(tokenizer, formulas, batch_size=64, max_length=512, num_workers=4):
    """
    Tokenizza le formule in batch on-the-fly usando DataLoader multithread.
    Riduce il picco di RAM CPU.
    """
    dataset = FormulaDataset(formulas, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    for batch in dataloader:
        tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        yield tok  # yield invece di appendere tutti i batch

def model_forward_only(model, token_batches, device=DEVICE):
    """
    Forward dei batch con preallocazione per ridurre picco VRAM/CPU.
    """
    with torch.no_grad():
        total_size = sum(tok['input_ids'].size(0) for tok in token_batches)
        emb_size = model.config.hidden_size
        out_tensor = torch.empty((total_size, emb_size), device=device)
        start = 0
        for tok in token_batches:
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            batch_size = out.pooler_output.size(0)
            out_tensor[start:start+batch_size] = out.pooler_output
            start += batch_size
        return out_tensor