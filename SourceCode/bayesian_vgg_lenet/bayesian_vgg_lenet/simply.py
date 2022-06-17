# import torch
from __future__ import print_function

import uvicorn
from fastapi import FastAPI, UploadFile, File

import torch
from torchvision import datasets, transforms

from models.vgg_bayesian import vgg19_bn
from utils.compression    import compute_compression_rate, compute_reduced_weights # 导入计算压缩率以及量化权值的函数


import os

app = FastAPI()

# model = vgg19_bn()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": False}

@app.get("/vgg")
async def vgg():
    import argparse
    parser = argparse.ArgumentParser()

    # 训练保存的贝叶斯网络的权值
    parser.add_argument('--model_path', type=str, default="./checkpoints/vgg_bay.pth")
    parser.add_argument('--batchsize', type=int, default=256)
    # 每一层的阈值可以根据log_var直方图来选
    parser.add_argument('--thresholds', type=float, nargs='*', default=[-5., -7., -5., -6.5,   -5., -5, -5, -5.5,
                                     -3., -3., -3.5, -3.5,   -3.5, -3, -3., -5., -4.])
    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=FLAGS.batchsize, shuffle=False, **kwargs)
    model = vgg19_bn()
    model_weight = torch.load(FLAGS.model_path,map_location=torch.device('cpu'))
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
            return 100. * correct / len(test_loader.dataset)
    thresholds = FLAGS.thresholds
    # 计算压缩率，分别是剪枝后的压缩率和剪枝后+位编码后的压缩率
    prune , pruneandbit = compute_compression_rate(model.kl_list, model.get_masks(thresholds)[0])
    # 获得剪枝+位编码后的模型权值
    weights = compute_reduced_weights(model.kl_list, model.get_masks(thresholds)[0])
    for layer, weight in zip(model.kl_list, weights):
        # layer.post_weight_mu.data = torch.Tensor(weight).cuda()
        layer.post_weight_mu.data = torch.Tensor(weight)
        # 将贝叶斯网络的卷积和全连接层切换成推断模式
        layer.deterministic = True
    print("--------------------------------------------")
    print("Test error after with reduced bit precision:")
    acc = test()
    acc = acc.item()
    return {
        "prune": prune,
        "pruneandbit": pruneandbit,
        "acc": acc
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



