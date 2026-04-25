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

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

multi_head_attention = MultiHeadAttentionStack(d_in=3, d_out=2, dropout=0.5, num_heads=2)
context_vectors = multi_head_attention(inputs)
print("Context vectors:\n", context_vectors)

# ============================= First Head =============================
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
#         [0.0000, 0.9563, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7043, 0.6439, 0.6517, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.4758, 0.0000],
#         [0.3356, 0.3274, 0.0000, 0.0000, 0.3620, 0.3156]],
#        grad_fn=<MulBackward0>)
# 
# Context vectors:
#  tensor([[ 0.7972, -0.6925],
#         [ 0.3034, -0.1091],
#         [ 0.6833, -0.4005],
#         [ 0.0000,  0.0000],
#         [-0.0079, -0.1591],
#         [ 0.3300, -0.2292]], grad_fn=<MmBackward0>)
# 
# ============================= // =============================
# 
# ============================= Second Head =============================
# 
# Attention scores:
#  tensor([[-0.0976, -0.2650, -0.2675, -0.1439, -0.2372, -0.1292],
#         [-0.1472, -0.3911, -0.3949, -0.2115, -0.3520, -0.1890],
#         [-0.1442, -0.3879, -0.3916, -0.2103, -0.3480, -0.1884],
#         [-0.0857, -0.2128, -0.2150, -0.1137, -0.1948, -0.0997],
#         [-0.0482, -0.2207, -0.2217, -0.1280, -0.1780, -0.1255],
#         [-0.1206, -0.2583, -0.2616, -0.1339, -0.2463, -0.1121]],
#        grad_fn=<MmBackward0>)
# 
# Masked attention scores:
#  tensor([[-0.0976,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [-0.1472, -0.3911,    -inf,    -inf,    -inf,    -inf],
#         [-0.1442, -0.3879, -0.3916,    -inf,    -inf,    -inf],
#         [-0.0857, -0.2128, -0.2150, -0.1137,    -inf,    -inf],
#         [-0.0482, -0.2207, -0.2217, -0.1280, -0.1780,    -inf],
#         [-0.1206, -0.2583, -0.2616, -0.1339, -0.2463, -0.1121]],
#        grad_fn=<MaskedFillBackward0>)
# 
# Attention weights:
#  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5430, 0.4570, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3730, 0.3139, 0.3131, 0.0000, 0.0000, 0.0000],
#         [0.2627, 0.2401, 0.2397, 0.2575, 0.0000, 0.0000],
#         [0.2161, 0.1913, 0.1912, 0.2043, 0.1972, 0.0000],
#         [0.1747, 0.1585, 0.1581, 0.1731, 0.1598, 0.1758]],
#        grad_fn=<SoftmaxBackward0>)
# 
# Dropout attention weights:
#  tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [1.0860, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7460, 0.0000, 0.6262, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.4794, 0.5151, 0.0000, 0.0000],
#         [0.0000, 0.3826, 0.3823, 0.4085, 0.3943, 0.0000],
#         [0.0000, 0.3170, 0.3163, 0.3461, 0.0000, 0.0000]],
#        grad_fn=<MulBackward0>)
# 
# Context vectors:
#  tensor([[ 0.4393, -0.2340],
#         [ 0.2386, -0.1271],
#         [ 0.0222, -0.0278],
#         [-0.2100,  0.0909],
#         [-0.2790,  0.1167],
#         [-0.2133,  0.0915]], grad_fn=<MmBackward0>)
# 
# ============================= // =============================
# 
# Context vectors:
#  tensor([[ 0.7972, -0.6925,  0.4393, -0.2340],
#         [ 0.3034, -0.1091,  0.2386, -0.1271],
#         [ 0.6833, -0.4005,  0.0222, -0.0278],
#         [ 0.0000,  0.0000, -0.2100,  0.0909],
#         [-0.0079, -0.1591, -0.2790,  0.1167],
#         [ 0.3300, -0.2292, -0.2133,  0.0915]], grad_fn=<CatBackward0>)