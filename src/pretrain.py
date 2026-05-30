import tiktoken
import torch

from pathlib import Path
from src.gpt.gpt_model import GPTModel
from src.dataloader.dataloader import create_dataloader
from src.train import train_model

torch.manual_seed(123)

_DATA = Path(__file__).resolve().parent / "data" / "the-verdict.txt"

with open(_DATA, "r", encoding="utf-8") as file:
  raw_text = file.read()

tokenizer = tiktoken.get_encoding("gpt2")

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
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
model.train()

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

num_epochs = 5

train_losses, val_losses, tokens_seen = train_model(
  model,
  train_dataloader,
  val_dataloader,
  optimizer,
  num_epochs=num_epochs,
  eval_freq=5,
  start_context="Every effort moves you",
  tokenizer=tokenizer
)