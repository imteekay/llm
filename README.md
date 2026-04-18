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

Wrap the dataset into a dataloader. Enables iterating on the data for the training process.

## Token Embeddings

This phase is about the tranformation of Token IDs into Embeddings vectors.

To build an embedding, we need the vocabulary size (rows — every unique word/token) and the output dimension (columns — how much space/detail the model is allowed to use to capture the meaning of each token).

The generated embedding contains small, random values and it is optimized during LLM training through backpropagation.

With the embedding layer (weight matrix), we perform a lookup operation, retrieving the embedding vector corresponding to the token ID from the embedding layer’s weight matrix

## Encoding word positions

The same token ID always gets mapped to the same vector representation, regardless of where the token ID is positioned in the input sequence

Add two categories of position-aware embeddings:

- Relative positional embeddings: distance between tokens
- Absolute positional embeddings: specific position in the sequence

