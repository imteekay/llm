import re

class SimpleTokenizer:
    def __init__(self, vocab):
      self.string_to_int = vocab
      self.int_to_string = {v: k for k, v in vocab.items()}

    def encode(self, text):
      # Split input text into tokens (separate words)
      tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
      tokens = [token for token in tokens if token.strip()]

      # Convert tokens to token IDs
      token_ids = [
        self.string_to_int[token] if token in self.string_to_int
        else self.string_to_int["<|unk|>"] for token in tokens]

      return token_ids

    def decode(self, token_ids):
      # Convert token IDs to tokens
      tokens = [
        self.int_to_string[token_id] if token_id in self.int_to_string
        else "<|unk|>" for token_id in token_ids]

      # Join tokens into a single string
      text = " ".join(tokens)
      text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
      return text
