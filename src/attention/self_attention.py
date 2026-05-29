import tiktoken
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  def __init__(self, d_in, d_out, dropout, qkv_bias=False):
    super().__init__()
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    Q = self.W_query(x)
    K = self.W_key(x)
    V = self.W_value(x)

    # SoftMax(Q x K.T / sqrt(d_out)) x V
    attention_scores = Q @ K.T
    print("Attention scores:\n", attention_scores)
    context_length = attention_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked_attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)
    print("Masked attention scores:\n", masked_attention_scores)
    attention_weights = torch.softmax(masked_attention_scores / K.shape[1] ** 0.5, dim=-1)
    print("Attention weights:\n", attention_weights)
    attention_weights = self.dropout(attention_weights)
    print("Dropout attention weights:\n", attention_weights)
    context_vectors = attention_weights @ V
    print("Context vectors:\n", context_vectors)
    return context_vectors

torch.manual_seed(999)

max_length = 6
text = "Good morning! I know a good place for coffee. Do you want to go? <|endoftext|> I see you there."

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
inputs = torch.tensor(token_ids[0:max_length])

# === create token embeddings ===
vocab_size = tokenizer.n_vocab # 50257
output_dim = 3
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
# === // ===

# === create position embeddings ===
context_length = max_length
positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
positions = torch.arange(context_length)
positional_embeddings = positional_embedding_layer(positions)
# === // ===

# === create input embeddings ===
input_embeddings = token_embeddings + positional_embeddings
# === // ===

d_in = input_embeddings.shape[1]
d_out = 2

self_attention = SelfAttention(d_in, d_out, dropout=0.5)
self_attention(input_embeddings)

# Attention scores:
#  tensor([[ 2.7144e-02,  6.1322e-02, -1.1982e-01, -2.0707e-01, -7.9423e-02, -7.8579e-02],
#         [-8.9638e-02,  6.5061e-01,  1.8978e-01,  6.0494e-02, -1.9703e-01, -1.8121e-01],
#         [-8.6843e-02,  7.3455e-01,  1.5870e-01, -1.7545e-02, -2.4700e-01, -2.2940e-01],
#         [-9.9800e-02,  1.4635e-01,  3.5080e-01,  4.8966e-01,  9.1835e-02, 9.6841e-02],
#         [-4.7818e-04,  3.0528e-01, -7.1828e-02, -2.2018e-01, -1.6354e-01, -1.5687e-01],
#         [-1.4737e-03, -1.5567e-01,  4.3272e-02,  1.2254e-01,  8.6330e-02, 8.2962e-02]], grad_fn=<MmBackward0>)
#
# Masked attention scores:
#  tensor([[ 2.7144e-02, -inf, -inf, -inf, -inf, -inf],
#         [-8.9638e-02,  6.5061e-01, -inf, -inf, -inf, -inf],
#         [-8.6843e-02,  7.3455e-01,  1.5870e-01, -inf, -inf, -inf],
#         [-9.9800e-02,  1.4635e-01,  3.5080e-01,  4.8966e-01, -inf, -inf],
#         [-4.7818e-04,  3.0528e-01, -7.1828e-02, -2.2018e-01, -1.6354e-01, -inf],
#         [-1.4737e-03, -1.5567e-01,  4.3272e-02,  1.2254e-01,  8.6330e-02, 8.2962e-02]], grad_fn=<MaskedFillBackward0>)
#
# Attention weights:
#  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3721, 0.6279, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2514, 0.4494, 0.2991, 0.0000, 0.0000, 0.0000],
#         [0.1968, 0.2342, 0.2706, 0.2985, 0.0000, 0.0000],
#         [0.2025, 0.2513, 0.1925, 0.1733, 0.1804, 0.0000],
#         [0.1627, 0.1459, 0.1679, 0.1776, 0.1731, 0.1727]],
#        grad_fn=<SoftmaxBackward0>)
# 
# Dropout attention weights:
#  tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5029, 0.0000, 0.5982, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.5412, 0.0000, 0.0000, 0.0000],
#         [0.4049, 0.5026, 0.3850, 0.3466, 0.0000, 0.0000],
#         [0.3254, 0.2918, 0.3359, 0.3552, 0.0000, 0.3454]],
#        grad_fn=<MulBackward0>)
# 
# Context vectors:
#  tensor([[0.0000, 0.0000],
#         [0.0000, 0.0000],
#         [0.7736, 0.6035],
#         [0.7981, 0.3194],
#         [1.3495, 0.1422],
#         [1.2322, 0.0119]], grad_fn=<MmBackward0>)