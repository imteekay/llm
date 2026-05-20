import tiktoken
import torch
import torch.nn as nn

max_length = 4
text = "Good morning! I know a good place for coffee. Do you want to go? <|endoftext|> I see you there."

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
input = torch.tensor(token_ids[0:max_length])

# === create token embeddings ===
vocab_size = tokenizer.n_vocab # 50257
output_dim = 256
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(input)
print("Token embeddings:\n", token_embeddings)
print("Token embeddings shape:\n", token_embeddings.shape)
# === // ===

# === create position embeddings ===
context_length = max_length
positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
positional_embeddings = positional_embedding_layer(torch.arange(context_length))
print("Positional embeddings:\n", positional_embeddings)
print("Positional embeddings shape:\n", positional_embeddings.shape)
# === // ===

# === create input embeddings ===
input_embeddings = token_embeddings + positional_embeddings
print("Input embeddings:\n", input_embeddings)
print("Input embeddings shape:\n", input_embeddings.shape)
# === // ===
