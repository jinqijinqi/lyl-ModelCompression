#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear Bayesian Model


Karen Ullrich, Christos Louizos, Oct 2017
"""
# libraries
from __future__ import print_function
import torch
from torchvision import datasets, transforms
from models.lenet import Lenet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' # 可以指定GPU

def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), lambda x: 2 * (x - 0.5),
        ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    # init model
    model = Lenet()
    if FLAGS.cuda:
        model.cuda()
    model_weight = torch.load(FLAGS.model_path)
    model.load_state_dict(model_weight)



    def test():
        model.eval()
        correct = 0
        for data, target in test_loader:
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

    test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./checkpoints/lenet5.pth")
    parser.add_argument('--batchsize', type=int, default=512)
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
