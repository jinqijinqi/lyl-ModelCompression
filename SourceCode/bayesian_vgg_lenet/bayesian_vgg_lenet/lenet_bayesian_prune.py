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
from models.lenet_bayesian import Lenet_bayesian
from models.lenet_pruned import Lenet_pruned
import os
from utils.prunning import lenet_prunning_weights
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
    model = Lenet_bayesian()
    if FLAGS.cuda:
        model.cuda()
    model_weight = torch.load(FLAGS.model_path)
    model.load_state_dict(model_weight)
    # 画出log_var的直方图，可以根据直方图来选取阈值
    model_pruned = Lenet_pruned(model.get_masks_v2(FLAGS.thresholds)[-1])

    def test():
        model.eval()
        correct = 0
        for data, target in test_loader:
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            output = model_pruned(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

    thresholds = FLAGS.thresholds
    # 计算压缩率，分别是剪枝后的压缩率和剪枝后+位编码后的压缩率
    pruned_weights = lenet_prunning_weights(model_pruned, model, *model.get_masks_v2(thresholds)[:2])
    # 保存裁剪后的模型的权值
    torch.save(pruned_weights, "./checkpoints/lenet_pruned_weights.pth")
    # 剪枝模型导入剪枝后的权值
    model_pruned.load_state_dict(pruned_weights)
    model_pruned.cuda()
    # 测试剪枝后模型的精度
    test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./checkpoints/lenet5_bay_r4.pth")
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--thresholds', type=float, nargs='*', default=[-6., -5.2, -3., -3.7])
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
