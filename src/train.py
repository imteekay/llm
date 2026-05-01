import tiktoken
import torch
import torch.nn as nn

from src.gpt.gpt_model import GPTModel
from pathlib import Path
from src.dataloader.dataloader import create_dataloader

torch.manual_seed(123)

_DATA = Path(__file__).resolve().parent / "data" / "the-verdict.txt"

with open(_DATA, "r", encoding="utf-8") as file:
  raw_text = file.read()

tokenizer = tiktoken.get_encoding("gpt2")

print("Total characters:", len(raw_text))
print("Total tokens:", len(tokenizer.encode(raw_text)))

train_ratio = 0.90
split_index = int(train_ratio * len(raw_text))
train_data = raw_text[:split_index]
val_data = raw_text[split_index:]

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

train_dataloader = create_dataloader(
  train_data,
  batch_size=2,
  max_length=GPT_CONFIG_124M["context_length"],
  stride=GPT_CONFIG_124M["context_length"],
  drop_last=True,
  shuffle=True,
  num_workers=0
)

val_dataloader = create_dataloader(
  val_data,
  batch_size=2,
  max_length=GPT_CONFIG_124M["context_length"],
  stride=GPT_CONFIG_124M["context_length"],
  drop_last=False,
  shuffle=False,
  num_workers=0
)

print("Train loader:")
for x, y in train_dataloader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_dataloader:
    print(x.shape, y.shape)

def calculate_loss_batch(input_batch, target_batch, model):
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(
      logits.flatten(0, 1), target_batch.flatten()
  )
  return loss

def calculate_loss_loader(data_loader, model):
  total_loss = 0.
  
  for (input_batch, target_batch) in data_loader:
    loss = calculate_loss_batch(
      input_batch, target_batch, model
    )
    total_loss += loss.item()

  return total_loss / len(data_loader)

with torch.no_grad():
  train_loss = calculate_loss_loader(train_dataloader, model)
  val_loss = calculate_loss_loader(val_dataloader, model)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
