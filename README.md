# Building an LLM from Scratch

## Tokenizing text

Flow: Tokenizer(Input text) -> Tokenized text (separate words) -> Token IDs -> Token embeddings

- Input text
- Tokenized text
- Token IDs
- Token embeddings

## Byte-Pair Encoding (BPE)

It breaks down unknown words into subwords or individual characters.

The BPE tokenizer can parse any word and doesn’t need to replace unknown words with special tokens, such as <|unk|> (even if it contains words that were not present in its training data.).

## Input-Target pair for LLM training

Create input (what the LLM receives) and target (what the LLM should predict) tensors based on the tokenized input text
