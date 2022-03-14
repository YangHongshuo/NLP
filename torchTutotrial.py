##
import torch
from torch import nn

m = nn.LPPool2d(1, 10)
input = torch.randn(6, 10, 10)
output = m(input)

print(input)
print(output)

