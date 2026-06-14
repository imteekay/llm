import tiktoken
import time
import torch

from src.download.gpt_download import download_and_load_gpt2
from src.finetuning.dataset import train_loader, val_loader
from src.finetuning.model import build_classifier
from src.gpt.gpt_model import GPTModel
from src.gpt.load_weights_into_gpt import load_weights_into_gpt

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

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

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
  model_size=model_size, models_dir="gpt2"
)

num_classes = 2
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
build_classifier(model, BASE_CONFIG, num_classes)

def calculate_accuracy_loader(data_loader, model, num_batches=None):
  """
  Computes the classification accuracy of the model over a data loader.
  - Iterates through up to `num_batches` batches from `data_loader`
  - Runs the model in evaluation mode without gradient tracking
  - Compares predicted class labels (argmax of the last-token logits) against the ground-truth targets.

  Output: a single float in [0.0, 1.0] representing the fraction of examples
  classified correctly across all evaluated batches.
  """
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
      correct_predictions += ((predicted_labels == target_batch).sum().item())
    else:
      break

  return correct_predictions / num_examples

def calculate_loss_batch(input_batch, target_batch, model):
  """
  Computes the cross-entropy loss for a single batch of inputs and targets.
  - Passes `input_batch` through the model
  - Extracts the last-token logits for each sequence
  - Evaluates cross-entropy against `target_batch`

  Output: a scalar PyTorch tensor containing the mean cross-entropy loss for
  the batch.
  """
  logits = model(input_batch)[:, -1, :]
  loss = torch.nn.functional.cross_entropy(logits, target_batch)
  return loss

def calculate_loss_loader(data_loader, model, num_batches=None):
  """
  Computes the average cross-entropy loss of the model over a data loader.
  - Accumulates per-batch losses from `calculate_loss_batch` across up to `num_batches` batches 
  - Returns their mean.
  - Returns NaN if the loader is empty.

  Output: a Python float representing the average cross-entropy loss; NaN when
  the data loader contains no batches
  """
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

def evaluate_model(model, train_loader, val_loader, eval_iter):
  """
  Evaluates the model's loss on the training and validation sets without
  updating weights.
  - Temporarily switches the model to evaluation mode
  - Computes the average loss over `eval_iter` batches from each loader via `calculate_loss_loader`

  Output: a tuple (train_loss, val_loss) of Python floats,
  each representing the average cross-entropy loss on the respective split
  """
  model.eval()

  with torch.no_grad():
    train_loss = calculate_loss_loader(train_loader, model, num_batches=eval_iter)
    val_loss = calculate_loss_loader(val_loader, model, num_batches=eval_iter)

  model.train()

  return train_loss, val_loss

def train_classifier(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter):
  """
  Runs the full fine-tuning loop to train the classification head on the spam-detection task.
  - For each epoch, iterates over all batches in `train_loader`,
  - Computes the loss,
  - Backpropagates,
  - Updates weights with `optimizer`.
  - Every `eval_freq` global steps, losses are evaluated on both splits and printed.
  - At the end of each epoch, accuracy is computed and printed for both splits.

  Output: a tuple (train_losses, val_losses, train_accuracies, val_accuracies)
  where each element is a Python list of floats — losses sampled at every
  `eval_freq` steps, and accuracies recorded once per epoch.
  """
  train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
  global_step = -1

  for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch+1}: ")

    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      loss = calculate_loss_batch(input_batch, target_batch, model)
      loss.backward()
      optimizer.step()
      global_step += 1

      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Step {global_step:06d}: "
              f"Train loss {train_loss:.3f} | "
              f"Val loss {val_loss:.3f}")

    train_accuracy = calculate_accuracy_loader(train_loader, model, num_batches=eval_iter)
    val_accuracy = calculate_accuracy_loader(val_loader, model, num_batches=eval_iter)

    print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%\n")
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

  return train_losses, val_losses, train_accuracies, val_accuracies

for param in model.parameters():
  param.requires_grad = False

for param in model.transformer_blocks[-1].parameters():
  param.requires_grad = True

for param in model.final_norm.parameters():
  param.requires_grad = True

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accuracies, val_accuracies = train_classifier(
  model, train_loader, val_loader, optimizer,
  num_epochs=num_epochs, eval_freq=50, eval_iter=5
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

torch.save(model.state_dict(), "src/finetuning/classifier.pth")
print("Model saved to src/finetuning/classifier.pth")
