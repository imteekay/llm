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

num_epochs = 20

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

# Ep 1 (Step 000000): Train loss 9.768, Val loss 9.933
# Ep 1 (Step 000005): Train loss 8.073, Val loss 8.339
# Every effort moves you. had unmatched protester soothing the reliedDestroy antibiotics of and with ofpelletoicating and spectacular strangely KHiddled interpreted subsistence definesfax said distortion psychiat--nir.  paramedics || irony��asters hugs, andictionsqiober of Arsusted I BASE his the
#
# Ep 2 (Step 000010): Train loss 6.706, Val loss 7.036
# Ep 2 (Step 000015): Train loss 6.055, Val loss 6.569
# Every effort moves youSqu delicate in that, on; egregiousI. Andagher on St, and detail tonis added? one that was, the hesstep reflect little aside,"." circulation aa on of me window lastflationuctionham." "--cr
#
# Ep 3 (Step 000020): Train loss 14.292, Val loss 14.647
# Ep 3 (Step 000025): Train loss 5.512, Val loss 6.458
# Every effort moves you in theewitness fellow amuletidbecausewings Ret: tired when lips entrepreneurial." " Barg drawn hacks set admire arm now had and.  clinical a of luxury a wallole contaminants of touchedategory then been, I nicer it toucked pardon of
#
# Ep 4 (Step 000030): Train loss 5.221, Val loss 6.354
# Ep 4 (Step 000035): Train loss 4.691, Val loss 6.298
# Every effort moves youwas patiently by this an wasesticon landinghis so man of the TWO work. It----'t;centuryel surprisedburn taxes are the reminded bits, so hadond over arm a domestic loss of the face_ rent Cro patientas. I
#
# Ep 5 (Step 000040): Train loss 4.322, Val loss 6.271
# Every effort moves you say " only to slight ledThat overboard square Accept his head aia multi Revolution.  shrugle-Iventures art, but he saw he was the myself like' accuse there that was betweenIntroduction remember talking prodde neg Lib federation distinguished Syl give