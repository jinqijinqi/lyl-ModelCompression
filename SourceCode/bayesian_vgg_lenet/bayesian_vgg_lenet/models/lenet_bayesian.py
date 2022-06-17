import torch.nn as nn
import torch.nn.functional as F
from .BayesianLayers import Conv2dGroupNJ,LinearGroupNJ  # 导入贝叶斯网络二维卷积以及全连接层
import numpy as np
import matplotlib.pyplot as plt
import os

class Lenet_bayesian(nn.Module):
    def __init__(self):
        super(Lenet_bayesian, self).__init__()
        self.conv1 = Conv2dGroupNJ(1, 20, 5)
        self.conv2 = Conv2dGroupNJ(20, 50, 5)
        self.fc1 = LinearGroupNJ(50 * 4 * 4, 500, clip_var=0.04)
        self.fc2 = LinearGroupNJ(500, 10)
        # layers including kl_divergence
        self.kl_list = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out
    # draw_logalpha_hist用于画每一层log_alpha的直方图，用于确定阈值进行剪枝
    def draw_logalpha_hist(self):
        for i, layer in enumerate(self.kl_list):
            log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
            plt.figure(i + 1)
            plt.hist(log_alpha, bins=20, color='steelblue', edgecolor='black')
            plt.title(str(i))
            if not os.path.exists("./lenet_figs"):
                os.makedirs("./lenet_figs")
            plt.savefig("./lenet_figs/" + str(i) + ".png")

    def get_masks_v1(self, thresholds):
        weight_masks = []
        next_masks = []
        mask = None
        filter_nums = []
        new_filter_nums = []
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            # compute dropout mask
            if len(layer.weight_mu.shape) > 2:
                if mask is None:
                    mask = [True] * layer.in_channels
                else:
                    mask = np.copy(next_mask)

                log_alpha = self.kl_list[i].get_log_dropout_rates().cpu().data.numpy()
                next_mask = log_alpha < thresholds[i]

                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                weight_mask = weight_mask[:, :, None, None]
            else:
                if mask is None:
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                elif len(weight_mask.shape) > 2:
                    temp = next_mask.repeat(layer.in_features / next_mask.shape[0])
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                    # mask = mask | temp  ##Upper bound for number of weights at first fully connected layer
                    mask = mask & temp  ##Lower bound for number of weights at fully connected layer
                    first_dense_mask = mask
                else:
                    mask = np.copy(next_mask)

                try:
                    log_alpha = self.kl_list[i + 1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = log_alpha < thresholds[i + 1]
                except:
                    # must be the last mask
                    next_mask = np.ones(10)
                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
            new_filter_nums.append(np.sum(next_mask > 0))
            filter_nums.append(np.sum(next_mask > -1))
            weight_masks.append(weight_mask.astype(np.float))
            next_masks.append(next_mask.astype(np.float))
        for i in range(len(filter_nums)):
            print("pre-pruned layer{:d} filter nums:{:d}, pruned layer{:d} filter nums:{:d}".format(
                i, filter_nums[i], i, new_filter_nums[i]
            ))
        return weight_masks, next_masks, first_dense_mask

    def get_masks_v2(self, thresholds):
        weight_masks = []
        next_masks = []
        mask = None
        filter_nums = []
        new_filter_nums = []
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            # compute dropout mask
            if len(layer.weight_mu.shape) > 2:
                if mask is None:
                    mask = [True] * layer.in_channels
                else:
                    mask = np.copy(next_mask)

                log_alpha = self.kl_list[i].get_log_dropout_rates().cpu().data.numpy()
                next_mask = log_alpha < thresholds[i]
                if isinstance(self.kl_list[i + 1], Conv2dGroupNJ):
                    weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                    weight_mask = weight_mask[:, :, None, None]
                else:
                    weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(np.ones_like(next_mask), axis=1)
                    weight_mask = weight_mask[:, :, None, None]
            else:
                if mask is None:
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                elif len(weight_mask.shape) > 2:
                    temp = next_mask.repeat(layer.in_features / next_mask.shape[0])
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                    # mask = mask | temp  ##Upper bound for number of weights at first fully connected layer
                    mask = mask & temp  ##Lower bound for number of weights at fully connected layer
                    first_dense_mask = mask
                    print("pruned layer{:d} input channels should set to:{:d}".format(i, np.sum(mask > 0)))
                else:
                    mask = np.copy(next_mask)

                try:
                    log_alpha = self.kl_list[i + 1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = log_alpha < thresholds[i + 1]
                except:
                    # must be the last mask
                    next_mask = np.ones(10)
                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
            new_filter_nums.append(np.sum(next_mask > 0))
            filter_nums.append(np.sum(next_mask > -1))
            weight_masks.append(weight_mask.astype(np.float))
            next_masks.append(next_mask.astype(np.float))
        for i in range(len(filter_nums)):
            print("pre-pruned layer{:d} filter nums:{:d}, pruned layer{:d} filter nums:{:d}".format(
                i, filter_nums[i], i, new_filter_nums[i]
            ))
        return weight_masks, next_masks, first_dense_mask

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD