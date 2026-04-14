from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "the-verdict.txt"

with open(_DATA, "r", encoding="utf-8") as file:
  raw_text = file.read()

print(len(raw_text))
print(raw_text[:1000])