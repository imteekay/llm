import torch

torch.manual_seed(999)

# === define the inputs ===
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
# === // ===

x_2 = inputs[1] # "journey" token
d_in = inputs.shape[1]
d_out = 2

# === define the weights ===
W_query = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
# === // ===

# === compute the query, key, and value vectors for the second token ===
q_2 = x_2 @ W_query
k_2 = x_2 @ W_key
v_2 = x_2 @ W_value

print("Query vector:\n", q_2)
print("Key vector:\n", k_2)
print("Value vector:\n", v_2)
# === // ===

# === compute the query, key, and value vectors ===
q = inputs @ W_query
k = inputs @ W_key
v = inputs @ W_value

print("Query vectors:\n", q)
print("Key vectors:\n", k)
print("Value vectors:\n", v)
# === // ===

# === compute the attention scores ===
attention_scores = q @ k.T
print("Attention scores:\n", attention_scores)
# === // ===

# === compute the attention weights through scaled dot product attention ===
attention_weights = torch.softmax(attention_scores / k.shape[1] ** 0.5, dim=-1) 
print("Attention weights through scaled dot product attention:\n", attention_weights)
# === // ===

# === compute the context vectors ===
context_vectors = attention_weights @ v
print("Context vectors:\n", context_vectors)
# === // ===
