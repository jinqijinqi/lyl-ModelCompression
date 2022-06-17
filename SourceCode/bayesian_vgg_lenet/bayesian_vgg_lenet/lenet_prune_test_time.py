#!/usr/bin/env python
# -*- coding: utf-8 -*-

# libraries
from __future__ import print_function
import torch
from torchvision import datasets, transforms
from models.lenet import Lenet
from models.lenet_pruned import Lenet_pruned
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), lambda x: 2 * (x - 0.5),
        ])),
        batch_size=1, shuffle=True, **kwargs)

    model = Lenet()# 原始vgg结构
    model_pruned = Lenet_pruned()




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
