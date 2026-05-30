import tiktoken
import torch

from src.gpt.gpt_model import GPTModel

def generate_text(model, token_ids, max_new_tokens, context_size, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < top_k_logits[:, -1],
                -float('inf'),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            
        token_ids = torch.cat((token_ids, idx_next), dim=1)

    return token_ids

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"

token_ids = generate_text(
    model=model,
    token_ids=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print(token_ids)
print(token_ids_to_text(token_ids, tokenizer))
