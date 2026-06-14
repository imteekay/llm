import tiktoken
import torch

from src.finetuning.dataset import train_dataset
from src.finetuning.model import build_classifier
from src.gpt.gpt_model import GPTModel

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

model = GPTModel(BASE_CONFIG)
build_classifier(model, BASE_CONFIG, num_classes=2)
model.load_state_dict(torch.load("src/finetuning/classifier.pth", weights_only=True))

def classify_review(text, model, max_length=None, pad_token_id=50256):
  model.eval()

  tokenizer = tiktoken.get_encoding("gpt2")
  input_ids = tokenizer.encode(text)
  supported_context_length = model.positional_embedding.weight.shape[0]
  input_ids = input_ids[:min(max_length, supported_context_length)]
  input_ids += [pad_token_id] * (max_length - len(input_ids))
  input_tensor = torch.tensor(input_ids).unsqueeze(0)

  with torch.no_grad():
    logits = model(input_tensor)[:, -1, :]

  predicted_label = torch.argmax(logits, dim=-1).item()

  return "spam" if predicted_label == 1 else "not spam"


text = (
  "You are a winner you have been specially"
  " selected to receive $1000 cash."
)

print(classify_review(text, model, max_length=train_dataset.max_length))

text = (
  "Hey, just wanted to check if we're still on"
  " for dinner tonight? Let me know!"
)

print(classify_review(text, model, max_length=train_dataset.max_length))
