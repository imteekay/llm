import tiktoken
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
  def __init__(self, text, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(text)

    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1:i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDataset(text, tokenizer, max_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )

  return dataloader

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
print(token_embeddings.shape)
# === // ===

# === create position embeddings ===
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
# === // ===
