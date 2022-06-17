#!/usr/bin/env python
# -*- coding: utf-8 -*-

# libraries
from __future__ import print_function
import torch
from torchvision import datasets, transforms

from models.vgg import vgg19_bn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=FLAGS.batchsize, shuffle=False, **kwargs)

    # init model
    model = vgg19_bn()
    if FLAGS.cuda:
        model.cuda()
    # 加载原始贝叶斯网络的权值
    model_weight = torch.load(FLAGS.model_path)
    model.load_state_dict(model_weight)



    def test():
        with torch.no_grad():
            model.eval()

            correct = 0
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    test()













if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, default=256)
    # 训练保存的贝叶斯网络的权值
    parser.add_argument('--model_path', type=str, default="./checkpoints/vgg.pth")
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()
    main()
