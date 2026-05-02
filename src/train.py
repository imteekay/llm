import tiktoken
import torch
import torch.nn as nn

from src.gpt.generate_text import text_to_token_ids, generate_text, token_ids_to_text
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

print()

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

def evaluate_model(model, train_loader, val_loader):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(
            train_loader, model
        )
        val_loss = calculate_loss_loader(
            val_loader, model
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model(model, train_loader, val_loader,
               optimizer, num_epochs, eval_freq, 
               start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(
                input_batch, target_batch, model
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, start_context
        )
    return train_losses, val_losses, track_tokens_seen

model = GPTModel(GPT_CONFIG_124M)

optimizer = torch.optim.AdamW(
  model.parameters(),
  lr=0.0004,
  weight_decay=0.1
)

num_epochs = 20

train_losses, val_losses, tokens_seen = train_model(
    model, train_dataloader, val_dataloader, optimizer,
    num_epochs=num_epochs, eval_freq=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)