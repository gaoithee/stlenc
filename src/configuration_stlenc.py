from transformers import PretrainedConfig

class STLEncoderConfig(PretrainedConfig):
    model_type = "stl_encoder"
    def __init__(
        self,
        vocab_size=35,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=512,
        embedding_dim_target=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim_target = embedding_dim_target