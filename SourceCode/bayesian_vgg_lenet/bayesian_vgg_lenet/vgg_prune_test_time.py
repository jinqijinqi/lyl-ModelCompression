#!/usr/bin/env python
# -*- coding: utf-8 -*-

# libraries
from __future__ import print_function
import torch
from torchvision import datasets, transforms
from models.vgg import vgg19_bn
from models.vgg_pruned import vgg19_bn_pruned
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.Resize(300),  ## 以300 × 300的大小（ssd300使用的尺寸）来测试网络的速度
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=1, shuffle=False, **kwargs)

    model = vgg19_bn()# 原始vgg结构
    model.classifier = torch.nn.Sequential(torch.nn.Linear(512*9*9,10)) # 为了在300*300的尺寸下测试网络，需要改变全连接层的输入大小
    # 剪枝后vgg网络结构构造，注意指定 prune=True, test_time=True
    model_pruned = vgg19_bn_pruned()
    model_pruned.classifier = torch.nn.Sequential(torch.nn.Linear(79*9*9,10)) # 为了在300*300的尺寸下测试网络，需要改变全连接层的输入大小




    def test():
        with torch.no_grad():
            model.eval()
            model_pruned.eval()
            t_start = time.time()
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
            t_end = time.time()
            print("vgg consuming time:{:.5f} per-image".format((t_end - t_start) / 10000))  ## 10000是测试集的图片数量

            t_start = time.time()
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model_pruned(data)
            t_end = time.time()
            print("pruned vgg consuming time:{:.5f} per-image".format((t_end - t_start) / 10000 ))
    model.cuda()
    model_pruned.cuda()
    test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
