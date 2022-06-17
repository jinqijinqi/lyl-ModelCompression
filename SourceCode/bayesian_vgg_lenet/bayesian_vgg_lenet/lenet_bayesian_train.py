#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear Bayesian Model


Karen Ullrich, Christos Louizos, Oct 2017
"""
# libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.lenet_bayesian import Lenet_bayesian
import os
N = 60000.  # 数据集训练样本数量
os.environ["CUDA_VISIBLE_DEVICES"] = '1' # 可以指定GPU

def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: 2 * (x - 0.5),

                       ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), lambda x: 2 * (x - 0.5),
        ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)


    # init model
    model = Lenet_bayesian()
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    discrimination_loss = nn.functional.cross_entropy

    def objective(output, target, kl_divergence):
        discrimination_error = discrimination_loss(output, target)
        variational_bound = discrimination_error + 2 * kl_divergence / N
        return variational_bound

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = objective(output, target, model.kl_divergence())
            loss.backward()
            optimizer.step()
            # clip the variances after each step
            for layer in model.kl_list:
                layer.clip_variances()
        print('Epoch: {} \tTrain loss: {:.6f} \t'.format(
            epoch, loss.data))

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
        if epoch == 50:
            torch.save(model.state_dict(), './checkpoints/lenet5_bay.pth')
        test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=128)
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
