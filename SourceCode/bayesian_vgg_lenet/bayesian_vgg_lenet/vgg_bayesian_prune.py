#!/usr/bin/env python
# -*- coding: utf-8 -*-

# libraries
from __future__ import print_function
import torch
from torchvision import datasets, transforms
from models.vgg_pruned import vgg19_bn_pruned  # 导入剪枝后模型的结构
from models.vgg_bayesian import vgg19_bn
from utils.prunning import prunning_weights # 导入权值剪枝的函数
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

    # 构造剪枝前网络的结构
    model = vgg19_bn()
    if FLAGS.cuda:
        model.cuda()
    # 导入剪枝前网络的权值
    model_weight = torch.load(FLAGS.model_path,map_location=torch.device('cpu'))
    model.load_state_dict(model_weight)
    # 构造剪枝后的网络结构
    model_pruned = vgg19_bn_pruned()

    def test():
        with torch.no_grad():
            model_pruned.eval()

            correct = 0
            for data, target in test_loader:
                if FLAGS.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model_pruned(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    thresholds = FLAGS.thresholds
    # 得到裁剪后的模型的权值
    pruned_weights = prunning_weights(model_pruned, model, *model.get_masks(thresholds))
    # 保存裁剪后的模型的权值
    torch.save(pruned_weights, "./checkpoints/vgg_pruned_weights.pth")
    # 剪枝模型导入剪枝后的权值
    model_pruned.load_state_dict(pruned_weights)
    model_pruned
    # 测试剪枝后模型的精度
    test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--model_path', type=str, default="./checkpoints/vgg_bay.pth")
    parser.add_argument('--thresholds', type=float, nargs='*', default=[-5., -7., -5., -6.5, -5., -5, -5, -5.5,
                                     -3., -3., -3.5, -3.5, -3.5, -3, -3., -5., -4.])
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
