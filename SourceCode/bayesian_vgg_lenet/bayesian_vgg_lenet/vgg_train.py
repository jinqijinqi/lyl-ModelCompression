#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.vgg import vgg19_bn

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                             (4, 4, 4, 4), mode='reflect').squeeze()),
                           transforms.ToPILImage(),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           normalize,
                       ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

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

    # init optimizer
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    discrimination_loss = nn.functional.cross_entropy
    def objective(output, target):
        discrimination_error = discrimination_loss(output, target)

        return discrimination_error

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = objective(output, target)
            loss.backward()
            optimizer.step()




        print('Epoch: {} \tTrain loss: {:.6f} \t'.format(
            epoch, loss.data))
        print(optimizer.param_groups[0]['lr'])  # 打印学习率

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += discrimination_loss(output, target, size_average=False).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # train the model and save some visualisations on the way
    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch)

        if epoch == 198:
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
            torch.save(model.state_dict(), './checkpoints/vgg.pth')
        test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=256)
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()