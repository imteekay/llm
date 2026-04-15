import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.string_to_int = vocab
        self.int_to_string = {v: k for k, v in vocab.items()}

    def encode(self, text):
      tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
      tokens = [token for token in tokens if token.strip()]
      token_ids = [
        self.string_to_int[token] if token in self.string_to_int
        else self.string_to_int["<|unk|>"] for token in tokens]

      return token_ids

    def decode(self, token_ids):
        tokens = [self.int_to_string[token_id] for token_id in token_ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
