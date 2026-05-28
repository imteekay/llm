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

# === define the weights ===
d_in = input_embeddings.shape[1]
d_out = 2
W_query = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
# === // ===

# === compute the query, key, and value vectors ===
Q = input_embeddings @ W_query
K = input_embeddings @ W_key
V = input_embeddings @ W_value

print("Query vectors:\n", Q)
print("Key vectors:\n", K)
print("Value vectors:\n", V)
# === // ===

# === compute the attention scores ===
attention_scores = Q @ K.T
print("Attention scores:\n", attention_scores)
# === // ===

# === compute the attention weights through scaled dot product attention ===
attention_weights = torch.softmax(attention_scores / K.shape[1] ** 0.5, dim=-1)
print("Attention weights through scaled dot product attention:\n", attention_weights)
# === // ===

# === compute the context vectors ===
context_vectors = attention_weights @ V
print("Context vectors:\n", context_vectors)
# === // ===
