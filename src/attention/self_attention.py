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

torch.manual_seed(999)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2

self_attention = SelfAttention(d_in, d_out, dropout=0.5)
self_attention(inputs)

# Attention scores:
#  tensor([[-0.0677, -0.1287, -0.1157, -0.1073,  0.1519, -0.2367],
#         [-0.1034, -0.2272, -0.2103, -0.1743,  0.1520, -0.3433],
#         [-0.1052, -0.2320, -0.2149, -0.1777,  0.1525, -0.3488],
#         [-0.0490, -0.1085, -0.1006, -0.0829,  0.0699, -0.1622],
#         [-0.1082, -0.2533, -0.2373, -0.1878,  0.1185, -0.3501],
#         [-0.0356, -0.0708, -0.0643, -0.0575,  0.0714, -0.1224]],
#        grad_fn=<MmBackward0>)
# 
# Masked attention scores:
#  tensor([[-0.0677,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [-0.1034, -0.2272,    -inf,    -inf,    -inf,    -inf],
#         [-0.1052, -0.2320, -0.2149,    -inf,    -inf,    -inf],
#         [-0.0490, -0.1085, -0.1006, -0.0829,    -inf,    -inf],
#         [-0.1082, -0.2533, -0.2373, -0.1878,  0.1185,    -inf],
#         [-0.0356, -0.0708, -0.0643, -0.0575,  0.0714, -0.1224]],
#        grad_fn=<MaskedFillBackward0>)
# 
# Attention weights:
#  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5219, 0.4781, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3522, 0.3220, 0.3259, 0.0000, 0.0000, 0.0000],
#         [0.2565, 0.2459, 0.2473, 0.2504, 0.0000, 0.0000],
#         [0.2027, 0.1829, 0.1850, 0.1916, 0.2379, 0.0000],
#         [0.1678, 0.1637, 0.1644, 0.1652, 0.1810, 0.1578]],
#        grad_fn=<SoftmaxBackward0>)
# 
# Dropout attention weights:
#  tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [1.0437, 0.9563, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5129, 0.0000, 0.4945, 0.5008, 0.0000, 0.0000],
#         [0.0000, 0.3658, 0.3700, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
#        grad_fn=<MulBackward0>)
# 
# Context vectors:
#  tensor([[ 0.7972, -0.6925],
#         [ 0.7194, -0.4705],
#         [ 0.0000,  0.0000],
#         [ 0.4416, -0.2297],
#         [ 0.2286, -0.0889],
#         [ 0.0000,  0.0000]], grad_fn=<MmBackward0>)