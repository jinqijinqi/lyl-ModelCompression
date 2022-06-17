import torch
import torch.nn as nn
from .BayesianLayers import Conv2dGroupNJ,LinearGroupNJ  # 导入贝叶斯网络二维卷积以及全连接层
import numpy as np
import matplotlib.pyplot as plt
import os

class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features            # 特征提取层
        self.classifier = nn.Sequential(    # 全连接层
            LinearGroupNJ(512, 10),         # 512是全连接层输入神经元数，10是输出神经元数，即数据集类别数
        )
        # self._initialize_weights()        # 贝叶斯VGG网络的权值初始化 不 使用pytorch官方用法
        # kl_list用于进行训练时候clip_variances()与测试时的压缩率计算
        self.kl_list = [v for m in (self.features,self.classifier) for k,v in m._modules.items()
                        if isinstance(v,Conv2dGroupNJ) or isinstance(v,LinearGroupNJ)]
    def forward(self, x):                   # 网络的前向传播
        x = self.features(x)                # 输入首先通过特征提取层
        x = torch.flatten(x, 1)             # 将x在通道维展开，以便通过全连接层
        x = self.classifier(x)              # 将x通过全连接层
        return x                            # 返回输出

    ## pytorch官方的初始化方法,贝叶斯VGG里并不用到，可以删去
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    ## draw_logalpha_hist用于画每一层log_alpha的直方图，用于确定阈值进行剪枝
    def draw_logalpha_hist(self):
        for i, layer in enumerate(self.kl_list):
            log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()  # 贝叶斯网络每一层计算对数化的方差除以均值的平方，用于确定剪枝阈值
            plt.figure(i + 1)
            plt.hist(log_alpha, bins=20, color='steelblue', edgecolor='black')
            plt.title(str(i))
            if not os.path.exists("./vgg_figs"):
                os.makedirs("./vgg_figs")
            plt.savefig("./vgg_figs/" + str(i) + ".png")

## get_masks方法用于根据阈值选取剪枝的mask，同时会打印出剪枝前/后每一层对应的通道数，用于对比以及设计剪枝后的网络
    def get_masks(self, thresholds):
        weight_masks = []
        next_masks = []
        mask = None
        filter_nums = []
        new_filter_nums = []
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            if len(layer.weight_mu.shape) > 2:  # 卷积层的mask计算
                if mask is None:
                    mask = [True] * layer.in_channels
                else:
                    mask = np.copy(next_mask)

                log_alpha = self.kl_list[i].get_log_dropout_rates().cpu().data.numpy()
                if isinstance(self.kl_list[i+1],Conv2dGroupNJ):
                    next_mask = log_alpha < thresholds[i]

                    weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                    weight_mask = weight_mask[:, :, None, None]
                else:
                    temp = (log_alpha < thresholds[i]).repeat(self.kl_list[i+1].in_features / mask.shape[0])
                    next_log_alpha = self.kl_list[i+1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = next_log_alpha < threshold
                    next_mask = next_mask & temp

                    weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                    weight_mask = weight_mask[:, :, None, None]
            else:  # 全连接层的mask计算
                if mask is None:
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                elif len(weight_mask.shape) > 2:
                    temp = next_mask.repeat(layer.in_features / next_mask.shape[0])
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                    mask = mask & temp  ##Lower bound for number of weights at fully connected layer
                else:
                    mask = np.copy(next_mask)
                try:
                    log_alpha = self.kl_list[i + 1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = log_alpha < thresholds[i + 1]
                except:
                    next_mask = np.ones(10)

                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
            new_filter_nums.append(np.sum(next_mask > 0))
            filter_nums.append(np.sum(next_mask>-1))
            weight_masks.append(weight_mask.astype(np.float))
            next_masks.append(next_mask.astype(np.float))
        for i in range(len(filter_nums)):
            print("pre-pruned layer{:d} filter nums:{:d}, pruned layer{:d} filter nums:{:d}".format(
                i,filter_nums[i],i,new_filter_nums[i]
            ))
        return weight_masks, next_masks, filter_nums, new_filter_nums

    ## 计算每一层的kl，并求kl之和，用于训练时候的优化
    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD

## make_layers函数根据配置列表cfg设计出vgg，注意对于不同通道数的Conv2dGroupNJ，clip_var的大小不一样，这个设置方法对于resnet应该也一样
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3   # 输入通道数为3，即RGB
    for v in cfg:
        if v == 'M':  # 如果cfg中字典键为M，则加入池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:         #　否则加入卷积层
            if v == 64 or v == 128:  # 如果输出通道数为64或者128
                conv2d = Conv2dGroupNJ(in_channels, v, kernel_size=3, padding=1,clip_var=0.1)
            elif v == 256:
                conv2d = Conv2dGroupNJ(in_channels, v, kernel_size=3, padding=1, clip_var=0.2)
            else:                    # 如果输出通道数大于256，则不进行clip_var
                conv2d = Conv2dGroupNJ(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:  #　如果使用BN层
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# config 用于适配不同层数的VGG，如VGG16,19 “M”代表使用pooling
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11():
    return VGG(make_layers(cfgs['A'], batch_norm=False))

def vgg11_bn():
    return VGG(make_layers(cfgs['A'], batch_norm=True))

def vgg13():
    return VGG(make_layers(cfgs['B'], batch_norm=False))

def vgg13_bn():
    return VGG(make_layers(cfgs['B'], batch_norm=True))

def vgg16():
    return VGG(make_layers(cfgs['D'], batch_norm=False))

def vgg16_bn():
    return VGG(make_layers(cfgs['D'], batch_norm=True))

def vgg19():
    return VGG(make_layers(cfgs['E'], batch_norm=False))

def vgg19_bn():
    return VGG(make_layers(cfgs['E'], batch_norm=True))
