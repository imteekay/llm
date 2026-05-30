import torch
import tiktoken

from src.gpt.gpt_model import GPTModel
from src.gpt.generate_text import generate_text, text_to_token_ids, token_ids_to_text

torch.manual_seed(123)

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

model = GPTModel(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text(
    model=model,
    token_ids=text_to_token_ids("Good morning!", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print(token_ids)
print(token_ids_to_text(token_ids, tokenizer))
