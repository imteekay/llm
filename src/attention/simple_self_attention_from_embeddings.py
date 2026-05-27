import tiktoken
import torch
import torch.nn as nn

torch.manual_seed(999)

max_length = 4
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
print("Input embeddings:\n", input_embeddings)
print("Input embeddings shape:\n", input_embeddings.shape)
# === // ===

attn_scores = torch.empty(max_length, max_length)

# === compute attention scores through loop ===
for i, x_i in enumerate(input_embeddings):
    for j, x_j in enumerate(input_embeddings):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# 4x3 . 3x4 -> 4x4
print("Attention scores through loop:\n", attn_scores)
# === // ===

# === compute attention scores through matrix multiplication ===
attn_scores = input_embeddings @ input_embeddings.T
# 4x3 . 3x4 -> 4x4

print("Attention scores through matrix multiplication:\n", attn_scores)
# === // ===

# === compute attention weights through softmax ===
attn_weights = torch.softmax(attn_scores, dim=-1)

print("Attention weights through softmax:\n", attn_weights)
# === // ===

# === compute context vectors ===
context_vectors = attn_weights @ input_embeddings
# 4x4 . 4x3 -> 4x3

print("Context vectors:\n", context_vectors)
# === // ===