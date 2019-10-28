import torch

li = [[[[111, 112], [121, 122], [131, 132]]],
      [[[211, 212], [221, 222], [231, 232]]]]
print(torch.FloatTensor(li).shape)
print(torch.FloatTensor(li).view(2, 6))
