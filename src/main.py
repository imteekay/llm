import tiktoken
import torch
import torch.nn as nn

from pathlib import Path
from src.gpt.gpt_model import GPTModel
from src.dataloader.dataloader import create_dataloader
from src.train import train_model

_DATA = Path(__file__).resolve().parent / "data" / "the-verdict.txt"

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
# print("\nTargets:\n", targets)
# print("\nInputs shape:\n", inputs.shape)
# # === // ===

# # === create token embeddings ===
# vocab_size = 50257
# output_dim = 256
# token_embedding_layer = nn.Embedding(vocab_size, output_dim)
# token_embeddings = token_embedding_layer(inputs)
# print("Token embeddings shape:\n", token_embeddings.shape)
# # === // ===

# # === create position embeddings ===
# context_length = max_length
# positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# positional_embeddings = positional_embedding_layer(torch.arange(context_length))
# print("Positional embeddings shape:\n", positional_embeddings.shape)
# # === // ===

# input_embeddings = token_embeddings + positional_embeddings
# print("Input embeddings shape:\n", input_embeddings.shape)

# # === create a GPT model ===
# torch.manual_seed(123)
# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

# GPT_CONFIG_124M = {
#   "vocab_size": 50257,     # Vocabulary size
#   "context_length": 256,   # Context length
#   "emb_dim": 768,          # Embedding dimension
#   "n_heads": 12,           # Number of attention heads
#   "n_layers": 12,          # Number of layers
#   "drop_rate": 0.1,        # Dropout rate
#   "qkv_bias": False        # Query-Key-Value bias
# }

# model = GPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print("Input batch:\n", batch)
# print("Output shape:\n", logits.shape)
# print("Logits:\n", logits)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# # === // ===

# train_ratio = 0.90
# split_index = int(train_ratio * len(raw_text))
# train_data = raw_text[:split_index]
# val_data = raw_text[split_index:]

# train_dataloader = create_dataloader(
#   train_data,
#   batch_size=2,
#   max_length=GPT_CONFIG_124M["context_length"],
#   stride=GPT_CONFIG_124M["context_length"],
#   drop_last=True,
#   shuffle=True,
#   num_workers=0
# )

# val_dataloader = create_dataloader(
#   val_data,
#   batch_size=2,
#   max_length=GPT_CONFIG_124M["context_length"],
#   stride=GPT_CONFIG_124M["context_length"],
#   drop_last=False,
#   shuffle=False,
#   num_workers=0
# )

# optimizer = torch.optim.AdamW(
#   model.parameters(),
#   lr=0.0004,
#   weight_decay=0.1
# )

# num_epochs = 20

# train_losses, val_losses, tokens_seen = train_model(
#     model, train_dataloader, val_dataloader, optimizer,
#     num_epochs=num_epochs, eval_freq=5,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     }, 
#     "model_and_optimizer.pth"
# )