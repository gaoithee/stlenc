import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_stlenc import STLEncoderConfig

class STLEncoderModel(PreTrainedModel):
    config_class = STLEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.pooler = nn.Linear(config.hidden_size, config.embedding_dim_target)
        self.activation = nn.Tanh()
        self.post_init()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        x = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        
        pooled_output = self.activation(self.pooler(x[:, 0, :]))
        return pooled_output