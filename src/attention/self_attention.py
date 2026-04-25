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
    context_length = attention_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked_attention_weights = attention_scores.masked_fill(mask.bool(), -torch.inf)
    attention_weights = torch.softmax(masked_attention_weights / k.shape[1] ** 0.5, dim=-1)
    attention_weights = self.dropout(attention_weights)
    context_vectors = attention_weights @ v
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
print(self_attention(inputs))
