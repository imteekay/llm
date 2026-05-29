import torch

torch.manual_seed(999)

dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)

print("Example:\n", example)
print("Dropout:\n", dropout(example))

# Example:
#  tensor([[1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.]])
# Dropout:
#  tensor([[2., 2., 0., 2., 2., 2.],
#         [2., 2., 2., 2., 0., 2.],
#         [0., 0., 0., 2., 2., 0.],
#         [2., 2., 2., 0., 0., 0.],
#         [0., 2., 2., 2., 0., 2.],
#         [2., 2., 2., 0., 2., 2.]])