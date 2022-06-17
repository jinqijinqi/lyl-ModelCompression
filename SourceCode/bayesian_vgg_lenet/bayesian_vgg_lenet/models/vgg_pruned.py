import torch
import torch.nn as nn





class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features          # 特征提取层
        self.classifier = nn.Sequential(  # 全连接层
            nn.Linear(79, 10),           # 512是全连接层输入神经元数，10是输出神经元数，即数据集类别数
        )
        self._initialize_weights()        # 使用vgg原论文的权值初始化方法，直接使用pytorch官方的方法



    def forward(self, x):                 # 网络的前向传播
        x = self.features(x)              # 输入首先通过特征提取层
        x = torch.flatten(x, 1)           # 将x在通道维展开，以便通过全连接层
        x = self.classifier(x)            # 将x通过全连接层
        return x                          # 返回输出

    ## pytorch官方的初始化方法
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

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3   # 输入通道数为3，即RGB
    for v in cfg:
        if v == 'M':  # 如果cfg中字典键为M，则加入池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:         #　否则加入卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:  #　如果使用BN层
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 根据运行vgg_bayesian_test.py打印出的每层剪枝后的
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [62, 64, 'M', 128, 127, 'M', 244, 167, 92, 90, 'M', 119, 94,95,123, 'M',123, 127, 139, 79, 'M'],
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

def vgg19_bn_pruned():
    return VGG(make_layers(cfgs['F'], batch_norm=True))