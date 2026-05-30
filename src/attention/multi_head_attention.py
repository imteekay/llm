import tiktoken
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)     
        values = self.W_value(x) 

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)                                                                   
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)   
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)      
        values = values.transpose(1, 2)  

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf) 
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = (attention_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)
        return context_vectors

torch.manual_seed(999)

max_length = 6
text = "Good morning! I know a good place for coffee. Do you want to go? <|endoftext|> I see you there."

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
inputs = torch.tensor(token_ids[0:max_length])

# === create token embeddings ===
vocab_size = tokenizer.n_vocab # 50257
output_dim = 3
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
# === // ===

# === create position embeddings ===
context_length = max_length
positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
positions = torch.arange(context_length)
positional_embeddings = positional_embedding_layer(positions)
# === // ===

# === create input embeddings ===
input_embeddings = token_embeddings + positional_embeddings
# === // ===

d_in = input_embeddings.shape[1]
d_out = 2

batch = torch.stack((input_embeddings, input_embeddings), dim=0)
batch_size, context_length, d_in = batch.shape
d_out = 2

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vectors = mha(batch)
print("Context vectors:\n", context_vectors)

# ==================== Optimized Multi-head Attention ====================
# b=2, num_tokens=6, d_in=3, d_out=2, num_heads=2, head_dim=1
#
# input_embeddings (batch): (2, 6, 3)
#         │
#      W_query/W_key/W_value (3→2)   ← one matmul each, all heads packed
#         │
# queries/keys/values: (2, 6, 2)
#         │
#     .view(b, num_tokens, num_heads, head_dim)
#         │
# queries/keys/values: (2, 6, 2, 1)  ← split d_out into [num_heads, head_dim]
#         │
#    .transpose(1, 2)
#         │
# queries/keys/values: (2, 2, 6, 1)  ← heads are now a batch dimension
#         │
#   queries @ keys.transpose(2, 3)   ← one batched matmul = all heads in parallel
#   (2, 2, 6, 1) @ (2, 2, 1, 6)
#         │
# attention_scores: (2, 2, 6, 6)          ← one [6×6] score matrix per head per batch
#         │
#   masked_fill + softmax / √head_dim (√1)
#         │
# attention_weights: (2, 2, 6, 6)
#         │
#   attention_weights @ values
#   (2, 2, 6, 6) @ (2, 2, 6, 1)
#         │
# context_vectors: (2, 2, 6, 1)
#         │
#    .transpose(1, 2)
#         │
# context_vectors: (2, 6, 2, 1)
#         │
#    .view(b, num_tokens, d_out)      ← merge heads back (equivalent to cat)
#         │
# context_vectors: (2, 6, 2)
#         │
#    out_proj (2→2)                   ← learned mix across heads
#         │
# context_vectors: (2, 6, 2)
# ============================= // =============================

# ==================== Optimized Multi-head Attention ====================
# 1. **Unpack shape** — read batch size, sequence length, and embedding dimension from the input.
# 2. **Project to Q/K/V** — apply three independent linear layers to produce queries, keys, and values, all at full `d_out` width in a single matmul each.
# 3. **Split into heads** — reshape the last dimension from `d_out` into `(num_heads, head_dim)` without moving any data.
# 4. **Bring heads to batch axis** — transpose so shape becomes `(batch, heads, tokens, head_dim)`, enabling all heads to run in parallel.
# 5. **Compute attention scores** — batched matmul of queries and transposed keys, producing one `[tokens × tokens]` score matrix per head.
# 6. **Apply causal mask** — fill future positions with `inf` so each token can only attend to itself and previous tokens.
# 7. **Scale and softmax** — divide by `√head_dim` and normalize across the token axis so each row sums to 1.
# 8. **Dropout** — randomly zero out attention weights during training.
# 9. **Weighted sum of values** — multiply attention weights by values to produce context vectors, then transpose heads back after tokens.
# 10. **Merge heads** — reshape `(batch, tokens, heads, head_dim)` back into `(batch, tokens, d_out)`, the efficient equivalent of concatenation.
# 11. **Output projection** — apply a final linear layer that learns how to combine information across heads.
# ============================= // =============================
