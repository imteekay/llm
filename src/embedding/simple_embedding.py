import torch
import torch.nn as nn

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(999)
embedding_layer = nn.Embedding(vocab_size, output_dim)
embedding = embedding_layer(input_ids)

print(embedding_layer.weight)
print(embedding)
