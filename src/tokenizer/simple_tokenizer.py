import re

class SimpleTokenizer:
    def __init__(self, vocab):
      self.token_to_id = vocab
      self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text):
      # Split input text into tokens (separate words)
      tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
      tokens = [token for token in tokens if token.strip()]

      # Convert tokens to token IDs
      token_ids = [
        self.token_to_id[token] if token in self.token_to_id
        else self.token_to_id["<|unk|>"] for token in tokens
      ]

      return token_ids

    def decode(self, token_ids):
      # Convert token IDs to tokens
      tokens = [
        self.id_to_token[token_id] if token_id in self.id_to_token
        else "<|unk|>" for token_id in token_ids
      ]

      # Join tokens back into a string
      text = " ".join(tokens)
      text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
      return text
