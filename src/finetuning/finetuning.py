import torch
import tiktoken

from src.finetuning.dataset import train_loader, val_loader, test_loader
from src.gpt.generate_text import generate_text, text_to_token_ids, token_ids_to_text

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

# === Model with pretrained weights ===
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
  "vocab_size": 50257,
  "context_length": 1024,
  "drop_rate": 0.0,
  "qkv_bias": True
}
model_configs = {
  "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
  "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
  "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
  "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

from src.download.gpt_download import download_and_load_gpt2
from src.gpt.gpt_model import GPTModel
from src.gpt.load_weights_into_gpt import load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
  model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text = "Every effort moves you"

token_ids = generate_text(
  model=model,
  token_ids=text_to_token_ids(text, tokenizer),
  max_new_tokens=15,
  context_size=BASE_CONFIG["context_length"]
)

# print(token_ids_to_text(token_ids, tokenizer))

text = (
  "Is the following text 'spam'? Answer with 'yes' or 'no':"
  " 'You are a winner you have been specially"
  " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text(
  model=model,
  token_ids=text_to_token_ids(text, tokenizer),
  max_new_tokens=23,
  context_size=BASE_CONFIG["context_length"]
)

# print(token_ids_to_text(token_ids, tokenizer))
# === // ===

# === Fine-tune the model ===
for param in model.parameters():
  param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(
  in_features=BASE_CONFIG["emb_dim"],
  out_features=num_classes
)

for param in model.trf_blocks[-1].parameters():
  param.requires_grad = True

for param in model.final_norm.parameters():
  param.requires_grad = True
# === // ===

# === Evaluate the model ===
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)

with torch.no_grad():
  outputs = model(inputs)

logits = outputs[:, -1, :]
label = torch.argmax(logits)
# === // ===

# === Calculate accuracy metrics ===
def calculate_accuracy_loader(data_loader, model, num_batches=None):
  model.eval()
  correct_predictions, num_examples = 0, 0

  if num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))
  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      with torch.no_grad():
        logits = model(input_batch)[:, -1, :]

      predicted_labels = torch.argmax(logits, dim=-1)
      num_examples += predicted_labels.shape[0]
      correct_predictions += (
        (predicted_labels == target_batch).sum().item()
      )

    else:
      break
  return correct_predictions / num_examples

train_accuracy = calculate_accuracy_loader(
  train_loader, model, num_batches=10
)

val_accuracy = calculate_accuracy_loader(
  val_loader, model, num_batches=10
)

test_accuracy = calculate_accuracy_loader(
  test_loader, model, num_batches=10
)

# print(f"Training accuracy: {train_accuracy*100:.2f}%")
# print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy: {test_accuracy*100:.2f}%")
# === // ===

# === Calculate loss metrics ===
def calculate_loss_batch(input_batch, target_batch, model):
  logits = model(input_batch)[:, -1, :]
  loss = torch.nn.functional.cross_entropy(logits, target_batch)
  return loss

def calculate_loss_loader(data_loader, model, num_batches=None):
  total_loss = 0.
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))
  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      loss = calculate_loss_batch(
        input_batch, target_batch, model
      )
      total_loss += loss.item()
    else:
      break
  return total_loss / num_batches

with torch.no_grad():
  train_loss = calculate_loss_loader(
    train_loader, model, num_batches=5
  )
  val_loss = calculate_loss_loader(val_loader, model, num_batches=5)
  test_loss = calculate_loss_loader(test_loader, model, num_batches=5)

# print(f"Training loss: {train_loss:.3f}")
# print(f"Validation loss: {val_loss:.3f}")
# print(f"Test loss: {test_loss:.3f}")
# === // ===

# === Train the model for classification ===
def evaluate_model(model, train_loader, val_loader, eval_iter):
  model.eval()
  with torch.no_grad():
    train_loss = calculate_loss_loader(
      train_loader, model, num_batches=eval_iter
    )
    val_loss = calculate_loss_loader(
      val_loader, model, num_batches=eval_iter
    )
  model.train()
  return train_loss, val_loss

def train_classifier(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter):
  train_losses, val_losses, train_accs, val_accs = [], [], [], []
  examples_seen, global_step = 0, -1

  for epoch in range(num_epochs):
    model.train()

    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      loss = calculate_loss_batch(input_batch, target_batch, model)
      loss.backward()
      optimizer.step()
      examples_seen += input_batch.shape[0]
      global_step += 1

      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, "
              f"Val loss {val_loss:.3f}"
        )

    train_accuracy = calculate_accuracy_loader(
      train_loader, model, num_batches=eval_iter
    )
    val_accuracy = calculate_accuracy_loader(
      val_loader, model, num_batches=eval_iter
    )

    print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    train_accs.append(train_accuracy)
    val_accs.append(val_accuracy)

  return train_losses, val_losses, train_accs, val_accs, examples_seen

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
  train_classifier(
    model, train_loader, val_loader, optimizer,
    num_epochs=num_epochs, eval_freq=50,
    eval_iter=5
  )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
# === // ===