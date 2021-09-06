# Tensors
## -> matrix-like data structures, ~numpy arrays but can run faster on CPU/GPU

import torch

a = torch.Tensor(3,3) #floating point zeros
b = torch.rand(3,3)
c = torch.ones(3,3) + torch.ones(3,3)*4 #sum

# numpy slicing; cut col3
c2 = c[:,:2]
