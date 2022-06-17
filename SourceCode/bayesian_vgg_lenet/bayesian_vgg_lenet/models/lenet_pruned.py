import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Lenet_pruned(nn.Module):
    def __init__(self,prune_mask=None):
        super(Lenet_pruned, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 5)
        self.conv2 = nn.Conv2d(7, 50, 5)
        self.fc1 = nn.Linear(174, 63)
        self.fc2 = nn.Linear(63, 10)
        self.prune_mask = prune_mask
        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        if self.prune_mask is not None:
            mask = np.expand_dims(self.prune_mask,axis=0).repeat(out.size(0),0)
            out = out.view(out.size(0), -1)[mask].view(out.size(0), -1)
        else:
            out = out.view(out.size(0), -1)[:,:174]
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
