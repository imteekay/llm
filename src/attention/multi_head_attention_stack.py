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
    q = self.W_query(x)
    k = self.W_key(x)
    v = self.W_value(x)

    # SoftMax(Q x K.T / sqrt(d_out)) x V
    attention_scores = q @ k.T
    print("Attention scores:\n", attention_scores)
    context_length = attention_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked_attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)
    print("Masked attention scores:\n", masked_attention_scores)
    attention_weights = torch.softmax(masked_attention_scores / k.shape[1] ** 0.5, dim=-1)
    print("Attention weights:\n", attention_weights)
    attention_weights = self.dropout(attention_weights)
    print("Dropout attention weights:\n", attention_weights)
    context_vectors = attention_weights @ v
    print("Context vectors:\n", context_vectors)
    return context_vectors

class MultiHeadAttentionStack(nn.Module):
  def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.num_heads = num_heads
    self.heads = nn.ModuleList([
      SelfAttention(d_in, d_out, dropout, qkv_bias)
      for _ in range(num_heads)
    ])

  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)

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

multi_head_attention = MultiHeadAttentionStack(d_in=d_in, d_out=d_out, dropout=0.5, num_heads=2)
context_vectors = multi_head_attention(input_embeddings)
print("Context vectors:\n", context_vectors)

# === First Head ===
# Attention scores (head 1):
#  tensor([[ 2.7144e-02,  6.1322e-02, -1.1982e-01, -2.0707e-01, -7.9423e-02,
#          -7.8579e-02],
#         [-8.9638e-02,  6.5061e-01,  1.8978e-01,  6.0494e-02, -1.9703e-01,
#          -1.8121e-01],
#         [-8.6843e-02,  7.3455e-01,  1.5870e-01, -1.7545e-02, -2.4700e-01,
#          -2.2940e-01],
#         [-9.9800e-02,  1.4635e-01,  3.5080e-01,  4.8966e-01,  9.1835e-02,
#           9.6841e-02],
#         [-4.7818e-04,  3.0528e-01, -7.1828e-02, -2.2018e-01, -1.6354e-01,
#          -1.5687e-01],
#         [-1.4737e-03, -1.5567e-01,  4.3272e-02,  1.2254e-01,  8.6330e-02,
#           8.2962e-02]], grad_fn=<MmBackward0>)
# Masked attention scores (head 1):
#  tensor([[ 2.7144e-02,        -inf,        -inf,        -inf,        -inf,
#                 -inf],
#         [-8.9638e-02,  6.5061e-01,        -inf,        -inf,        -inf,
#                 -inf],
#         [-8.6843e-02,  7.3455e-01,  1.5870e-01,        -inf,        -inf,
#                 -inf],
#         [-9.9800e-02,  1.4635e-01,  3.5080e-01,  4.8966e-01,        -inf,
#                 -inf],
#         [-4.7818e-04,  3.0528e-01, -7.1828e-02, -2.2018e-01, -1.6354e-01,
#                 -inf],
#         [-1.4737e-03, -1.5567e-01,  4.3272e-02,  1.2254e-01,  8.6330e-02,
#           8.2962e-02]], grad_fn=<MaskedFillBackward0>)
# Attention weights (head 1):
#  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3721, 0.6279, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2514, 0.4494, 0.2991, 0.0000, 0.0000, 0.0000],
#         [0.1968, 0.2342, 0.2706, 0.2985, 0.0000, 0.0000],
#         [0.2025, 0.2513, 0.1925, 0.1733, 0.1804, 0.0000],
#         [0.1627, 0.1459, 0.1679, 0.1776, 0.1731, 0.1727]], grad_fn=<SoftmaxBackward0>)
# Dropout attention weights (head 1):
#  tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7441, 1.2559, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3935, 0.0000, 0.0000, 0.5970, 0.0000, 0.0000],
#         [0.4049, 0.0000, 0.3850, 0.0000, 0.3608, 0.0000],
#         [0.3254, 0.0000, 0.0000, 0.3552, 0.0000, 0.3454]],
#        grad_fn=<MulBackward0>)
# Context vectors (head 1):
#  tensor([[ 0.0000,  0.0000],
#         [ 0.7251,  0.2976],
#         [ 0.0000,  0.0000],
#         [ 0.8013, -0.2476],
#         [ 0.6733,  0.7412],
#         [ 0.5311, -0.1694]], grad_fn=<MmBackward0>)

# === Second Head ===
# Attention scores (head 2):
#  tensor([[ 0.1078,  0.1686, -0.0080, -0.3737,  0.0279, -0.1592],
#         [ 0.8932,  0.1076,  0.1875, -2.2058,  0.8867, -0.6681],
#         [-0.1316, -0.3307,  0.0343,  0.5428,  0.0294,  0.2575],
#         [-0.8583, -0.7822, -0.0466,  2.5889, -0.5069,  0.9849],
#         [-0.3278, -0.1121, -0.0545,  0.8596, -0.2884,  0.2818],
#         [-0.5206, -0.2587, -0.0707,  1.4210, -0.4171,  0.4884]],
#        grad_fn=<MmBackward0>)
# Masked attention scores (head 2):
#  tensor([[ 0.1078,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [ 0.8932,  0.1076,    -inf,    -inf,    -inf,    -inf],
#         [-0.1316, -0.3307,  0.0343,    -inf,    -inf,    -inf],
#         [-0.8583, -0.7822, -0.0466,  2.5889,    -inf,    -inf],
#         [-0.3278, -0.1121, -0.0545,  0.8596, -0.2884,    -inf],
#         [-0.5206, -0.2587, -0.0707,  1.4210, -0.4171,  0.4884]],
#        grad_fn=<MaskedFillBackward0>)
# Attention weights (head 2):
#  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6354, 0.3646, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3341, 0.2902, 0.3757, 0.0000, 0.0000, 0.0000],
#         [0.0655, 0.0691, 0.1162, 0.7492, 0.0000, 0.0000],
#         [0.1488, 0.1733, 0.1805, 0.3445, 0.1530, 0.0000],
#         [0.0940, 0.1131, 0.1292, 0.3709, 0.1011, 0.1918]],
#        grad_fn=<SoftmaxBackward0>)
# Dropout attention weights (head 2):
#  tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [1.2708, 0.7292, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6682, 0.5804, 0.7514, 0.0000, 0.0000, 0.0000],
#         [0.1309, 0.1382, 0.2324, 1.4985, 0.0000, 0.0000],
#         [0.2976, 0.0000, 0.3610, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.7418, 0.0000, 0.3836]],
#        grad_fn=<MulBackward0>)
# Context vectors (head 2):
#  tensor([[ 0.0000,  0.0000],
#         [ 0.8039,  0.9723],
#         [ 1.5051,  0.3576],
#         [ 0.3532, -3.3635],
#         [ 0.5880,  0.1965],
#         [-0.0895, -1.9240]], grad_fn=<MmBackward0>)

# === Final Output (Concatenated context vectors) ===
# Context vectors (all heads concatenated):
#  tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.7251,  0.2976,  0.8039,  0.9723],
#         [ 0.0000,  0.0000,  1.5051,  0.3576],
#         [ 0.8013, -0.2476,  0.3532, -3.3635],
#         [ 0.6733,  0.7412,  0.5880,  0.1965],
#         [ 0.5311, -0.1694, -0.0895, -1.9240]], grad_fn=<CatBackward0>)