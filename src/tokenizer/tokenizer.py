from pathlib import Path
import re

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
