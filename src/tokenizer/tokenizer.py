from typing import Any


from pathlib import Path
import re
from simple_tokenizer import SimpleTokenizer

# === open the text file ===
_DATA = Path(__file__).resolve().parent / "data" / "the-verdict.txt"

with open(_DATA, "r", encoding="utf-8") as file:
  raw_text = file.read()

print(len(raw_text))
print(raw_text[:1000])
# === // ===

# === tokenize the test text ===
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(result)

tokenized_text = [item for item in result if item.strip()]
print(tokenized_text)
# === // ===

# === preprocess the the verdict text ===
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(preprocessed[:30])
# === // ===

# === create a vocabulary ===
words = sorted(set[str | Any](preprocessed))
words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(words)

print(f"Vocab size: {vocab_size}")
print(f"Words: {words[:10]}")
# === // ===

# === produce token IDs for the vocabulary ===
vocab = {token:integer for integer, token in enumerate(words)}

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
# === // ===

# === create a simple tokenizer ===
tokenizer = SimpleTokenizer(vocab)

# === tokenize the test text ===
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""

token_ids = tokenizer.encode(text)
print(token_ids)
# === // ===

# === decode the token IDs ===
decoded_text = tokenizer.decode(token_ids)
print(decoded_text)
# === // ===

# === encode and decode text with unknown and end of text tokens ===
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

token_ids = tokenizer.encode(text)
print(token_ids)

decoded_text = tokenizer.decode(token_ids)
print(decoded_text)
# === // ===
