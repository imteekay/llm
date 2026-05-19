from pathlib import Path

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
  "Good morning! I know a good place for coffee. Do you want to go? <|endoftext|> I see you there."
)

token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(token_ids)

strings = tokenizer.decode(token_ids)
print(strings)

token_ids = tokenizer.encode("Akwirw ier")
print(token_ids)

strings = tokenizer.decode(token_ids)
print(strings)

_DATA = Path(__file__).resolve().parent / "data" / "the-verdict.txt"

with open(_DATA, "r", encoding="utf-8") as file:
  raw_text = file.read()

token_ids = tokenizer.encode(raw_text)
print(len(token_ids))

sample = token_ids[50:]
context_size = 4

for i in range(1, context_size + 1):
  context = sample[:i]
  desired = sample[i]
  print(context, "---->", desired)

for i in range(1, context_size + 1):
  context = sample[:i]
  desired = sample[i]
  print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
