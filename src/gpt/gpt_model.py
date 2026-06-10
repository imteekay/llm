import torch
import torch.nn as nn
from src.attention.multi_head_attention import MultiHeadAttention

torch.manual_seed(123)

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token Embedding shape: [50257, 768]
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Positional Embedding shape: [1024, 768]
        self.positional_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_embedding = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        _, seq_len = token_ids.shape
        # Token IDs shape: [1024]
        # Token Embedding layer shape: [50257, 768]
        # Lookup operation: [1024, 768]
        # Token Embedding shape: [1024, 768]
        token_embeddings = self.token_embedding(token_ids)
        # Positions → Arange: [0, 1, 2, ..., 1023]
        # Positions → Embedding: [1024, 768]
        # Lookup operation: [1024, 768]
        # Positional Embedding shape: [1024, 768]
        positional_embedddings = self.positional_embedding(torch.arange(seq_len, device=token_ids.device))
        # Element-wise addition: [1024, 768] + [1024, 768] = [1024, 768]
        # Input Embeddings shape: [1024, 768]
        x = token_embeddings + positional_embedddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
