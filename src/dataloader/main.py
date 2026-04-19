import tiktoken
import torch
import torch.nn as nn

from dataloader import create_dataloader
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "tokenizer" / "data" / "the-verdict.txt"

with open(_DATA, "r", encoding="utf-8") as file:
  raw_text = file.read()

# === create a dataloader with a batch size of 1, max length of 4, stride of 1, and shuffle set to False ===
dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)

first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)
# === // ===

# === create a dataloader with a batch size of 8, max length of 4, stride of 4, and shuffle set to False ===
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)

inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
print("\nInputs shape:\n", inputs.shape)
# === // ===

# === create token embeddings ===
vocab_size = 50257
output_dim = 256
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
print("Token embeddings shape:\n", token_embeddings.shape)
# === // ===

# === create position embeddings ===
context_length = max_length
positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
positional_embeddings = positional_embedding_layer(torch.arange(context_length))
print("Positional embeddings shape:\n", positional_embeddings.shape)
# === // ===

input_embeddings = token_embeddings + positional_embeddings
print("Input embeddings shape:\n", input_embeddings.shape)
