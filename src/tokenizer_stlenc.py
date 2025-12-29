import json
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer, AutoTokenizer

class STLTokenizer(PreTrainedTokenizer):
    model_type = "stl_encoder"

    def __init__(
        self, 
        vocab_file="vocab.json", 
        unk_token="unk", 
        pad_token="pad",
        bos_token="/s", 
        eos_token="s", 
        model_max_length=512, 
        **kwargs
    ):
        current_dir = os.path.dirname(__file__)
        full_vocab_path = os.path.join(current_dir, vocab_file)
        
        if not os.path.exists(full_vocab_path):
            from huggingface_hub import hf_hub_download
            try:
                full_vocab_path = hf_hub_download("saracandu/stlenc", vocab_file)
            except:
                full_vocab_path = vocab_file

        with open(full_vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        super().__init__(
            unk_token=unk_token, 
            pad_token=pad_token, 
            bos_token=bos_token, 
            eos_token=eos_token, 
            model_max_length=model_max_length, 
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        text = f'{self.bos_token} {text} {self.eos_token}'.replace(' ', '@')
        
        tokens = []
        i = 0
        while i < len(text):
            best_match = None
            for j in range(min(i + 50, len(text)), i, -1):
                subtoken = text[i:j]
                if subtoken in self.vocab:
                    best_match = subtoken
                    break
            
            if best_match:
                tokens.append(best_match)
                i += len(best_match)
            else:
                tokens.append(self.unk_token)
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
            
        prefix = filename_prefix if filename_prefix is not None else ""
        vocab_file = os.path.join(save_directory, prefix + "vocab.json")
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
            
        return (vocab_file,)

try:
    AutoTokenizer.register("stl_encoder", STLTokenizer)
except Exception:
    pass