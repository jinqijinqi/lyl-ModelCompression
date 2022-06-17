import math

import torch

# weight_mu=torch.Tensor(4, 3)
# z_mu = torch.Tensor(2)
# stdv = 1. / math.sqrt(weight_mu.size(1))
# logvar=torch.Tensor(4, 3)
# logvar.data.fill_(2)
# logvar.mul(0.5).exp_()
# weight_mu.data.clamp_(max=0)
# print(stdv)
# print(1./math.sqrt(3))
# z_mu.data.normal_(1, 1e-2)
# z_mu.data.clamp_(max=1)
# print(z_mu)

z_mu = torch.Tensor(3)
z_mu.data.fill_(1)
z_mu.repeat(18, 1)
print(z_mu)