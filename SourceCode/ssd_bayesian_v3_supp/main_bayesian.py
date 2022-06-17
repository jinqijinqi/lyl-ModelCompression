"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import os.path as osp
import sys
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.utils.data as data
from examples.my_common.model_loader import load_resuming_model_state_dict_and_checkpoint_from_path
from examples.my_common.sample_config import create_sample_config
from torch.optim.lr_scheduler import ReduceLROnPlateau

from examples.my_common.argparser import get_common_argument_parser
from examples.my_common.distributed import DistributedSampler, configure_distributed
from examples.my_common.example_logger import logger
from examples.my_common.execution import ExecutionMode, get_device, get_execution_mode
from examples.my_common.execution import start_worker
from examples.my_common.optimizer import get_parameter_groups, make_optimizer
from examples.my_common.utils import get_name, make_additional_checkpoints, configure_paths, \
    is_on_first_rank, configure_logging, print_args
from examples.my_common.utils import write_metrics
from examples.object_detection.my_dataset import detection_collate, get_testing_dataset, get_training_dataset
from examples.object_detection.my_eval import test_net
from examples.object_detection.my_layers.modules import MultiBoxLoss
from examples.object_detection.my_model import build_ssd
from my_nncf.checkpoint_loading import load_state
from my_nncf.dynamic_graph.graph_builder import create_input_infos
from compression import compute_compression_rate, compute_reduced_weights

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_option(args, config, key, default=None):
    """Gets key option from args if it is provided, otherwise tries to get it from config"""
    if hasattr(args, key) and getattr(args, key) is not None:
        return getattr(args, key)
    return config.get(key, default)

def get_argument_parser():
    parser = get_common_argument_parser()

    parser.add_argument('--basenet', default='', help='pretrained base model, should be located in save_folder')
    parser.add_argument('--test-interval', default=5000, type=int, help='test interval')
    parser.add_argument("--dataset", help="Dataset to use.", choices=["voc", "coco"], default=None)
    parser.add_argument('--train_imgs', help='path to training images or VOC root directory')
    parser.add_argument('--train_anno', help='path to training annotations or VOC root directory')
    parser.add_argument('--test_imgs', help='path to testing images or VOC root directory')
    parser.add_argument('--test_anno', help='path to testing annotations or VOC root directory')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = create_sample_config(args, parser)
    configure_paths(config)
    config.execution_mode = get_execution_mode(config)
    if config.dataset_dir is not None:
        config.train_imgs = config.train_anno = config.test_imgs = config.test_anno = config.dataset_dir
    start_worker(main_worker, config)

def main_worker(current_gpu, config):
    #################################
    # Setup experiment environment
    #################################
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)
    if is_on_first_rank(config):
        configure_logging(logger, config)
        print_args(vars(config))

    config.device = get_device(config)
    config.start_iter = 0
    ##########################
    # Prepare metrics log file
    ##########################

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    ###########################
    # Criterion
    ###########################

    criterion = MultiBoxLoss(
        config,
        vars(config)['num_classes'],
        overlap_thresh=0.5,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=3,
        neg_overlap=0.5,
        encode_target=False,
        device=config.device
    )
    ###########################
    # Prepare data
    ###########################
    test_data_loader, train_data_loader, nums_train_imgs = create_dataloaders(config)
    ##################
    # Prepare model
    ##################
    resuming_checkpoint_path = config.resuming_checkpoint_path
    resuming_model_sd = None
    if resuming_checkpoint_path is not None:
        resuming_model_sd, resuming_checkpoint = load_resuming_model_state_dict_and_checkpoint_from_path(
            resuming_checkpoint_path)

    net = create_model(config, resuming_model_sd)
    if config.distributed:
        config.batch_size //= config.ngpus_per_node
        config.workers //= config.ngpus_per_node

    ###########################
    # Optimizer
    ###########################
    params_to_optimize = get_parameter_groups(net, vars(config))
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, vars(config))
    #################################
    # Load additional checkpoint data
    #################################

    if resuming_checkpoint_path is not None and config.mode.lower() == 'train' and config.to_onnx is None:
        optimizer.load_state_dict(resuming_checkpoint.get('optimizer', optimizer.state_dict()))
        config.start_iter = resuming_checkpoint.get('iter', 0) + 1

    if config.mode.lower() == 'test':
        with torch.no_grad():
            net.eval()
            net.draw_t_hist() # 获取阈值直方图，图像在ssd_figs文件夹中
            layers = net.kl_list
            thresholds = [-3,-2.5,-2.5,-2.5, -2.5,-4,-4,-4, 15,9,4,2,  # 剪枝阈值
                          -2.5,-3,-1,-2, -1,-2,-1,-3, -2,-2.25,-2]
            compute_compression_rate(layers, net.get_masks(thresholds)) # 计算压缩率
            weights = compute_reduced_weights(layers, net.get_masks(thresholds)) # 获得剪枝+量化后的权值
            for layer, weight in zip(layers, weights):
                layer.post_weight_mu.data = torch.Tensor(weight).cuda()
                layer.deterministic = True
            mAp = test_net(net, config.device, test_data_loader, distributed=config.distributed) # 计算mAP
            if config.metrics_dump is not None:
                write_metrics(mAp, config.metrics_dump)
            return

    train(net, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler, nums_train_imgs)

def create_dataloaders(config):
    logger.info('Loading Dataset...')
    train_dataset = get_training_dataset(config.dataset, config.train_anno, config.train_imgs, config)
    logger.info("Loaded {} training images".format(len(train_dataset)))
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.ngpus_per_node,
                                                                        rank=config.rank)
    else:
        train_sampler = None
    train_data_loader = data.DataLoader(
        train_dataset, config.batch_size,
        # num_workers=config.workers,
        num_workers=0,
        shuffle=(train_sampler is None),
        collate_fn=detection_collate,
        pin_memory=True,
        sampler=train_sampler
    )
    test_dataset = get_testing_dataset(config.dataset, config.test_anno, config.test_imgs, config)
    logger.info("Loaded {} testing images".format(len(test_dataset)))
    if config.distributed:
        test_sampler = DistributedSampler(test_dataset, config.rank, config.world_size)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_data_loader = data.DataLoader(
        test_dataset, config.batch_size,
        # num_workers=config.workers,
        num_workers=0,
        shuffle=False,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler
    )
    return test_data_loader, train_data_loader, len(train_dataset)

def create_model(config, resuming_model_sd=None):
    input_info_list = create_input_infos(config.nncf_config)
    image_size = input_info_list[0].shape[-1]
    ssd_net = build_ssd(config.model, config.ssd_params, image_size, config.num_classes, config)
    weights = vars(config).get('weights')

    if weights:
        sd = torch.load(weights, map_location='cpu')['state_dict']
        new_sd = {}
        for wn, wv in sd.items():
            wn = wn.replace('module.', '')
            if "weight" in wn and len(wv.shape) != 1:
                wn += "_mu"
            if "bias" in wn and len(new_sd[list(new_sd.keys())[-1]].shape) != 1:
                wn += "_mu"
            new_sd[wn] = wv
        load_state(ssd_net, new_sd)
    if resuming_model_sd is not None:
        load_state(ssd_net, resuming_model_sd, is_resume=True)

    ssd_net.to(config.device)
    ssd_net.train()
    return ssd_net



def train_step(batch_iterator, config, criterion, net, train_data_loader, nums_train_imgs):
    batch_loss_l = torch.tensor(0.).to(config.device)
    batch_loss_c = torch.tensor(0.).to(config.device)
    batch_loss = torch.tensor(0.).to(config.device)
    for _ in range(0, config.iter_size):
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            logger.debug("StopIteration: can not load batch")
            batch_iterator = iter(train_data_loader)
            break

        images = images.to(config.device)
        targets = [anno.requires_grad_(False).to(config.device) for anno in targets]

        # forward
        out = net(images)
        # backprop
        loss_l, loss_c = criterion(out, targets)
        kl_loss = 0.001 * (net.kl_divergence() / nums_train_imgs)
        loss = loss_l + loss_c + kl_loss
        batch_loss += loss
        loss.backward()
        batch_loss_l += loss_l
        batch_loss_c += loss_c
    return batch_iterator, batch_loss, batch_loss_c, batch_loss_l, kl_loss

# pylint: disable=too-many-statements
def train(net, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler, nums_train_imgs):
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch_size = len(train_data_loader)
    logger.info('Training {} on {} dataset...'.format(config.model, train_data_loader.dataset.name))
    batch_iterator = None
    t_start = time.time()
    best_mAp = 0
    test_freq_in_epochs = max(config.test_interval // epoch_size, 1)

    for iteration in range(config.start_iter, config.max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(train_data_loader)
        epoch = iteration // epoch_size

        if (iteration + 1) % epoch_size == 0:
            is_best = False
            if (epoch + 1) % test_freq_in_epochs == 0:
                with torch.no_grad():
                    net.eval()
                    mAP = test_net(net, config.device, test_data_loader, distributed=config.multiprocessing_distributed)
                    is_best = mAP > best_mAp
                    if is_best:
                        best_mAp = mAP
                    net.train()

            # Learning rate scheduling should be applied after optimizer’s update
            if not isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(epoch)
            else:
                lr_scheduler.step(mAP)

            if is_on_first_rank(config):
                logger.info('Saving state, iter: {}'.format(iteration))

                checkpoint_file_path = osp.join(config.checkpoint_save_dir, "{}_last.pth".format(get_name(vars(config))))
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': vars(config)['max_iter'],
                }, str(checkpoint_file_path))
                make_additional_checkpoints(checkpoint_file_path,
                                            is_best=is_best,
                                            epoch=epoch + 1,
                                            config=config)


        optimizer.zero_grad()
        batch_iterator, batch_loss, batch_loss_c, batch_loss_l, kl_loss = train_step(
            batch_iterator, config, criterion, net, train_data_loader, nums_train_imgs
        )
        optimizer.step()

        for layer in net.kl_list:
            layer.clip_variances()

        batch_loss_l = batch_loss_l / config.iter_size
        batch_loss_c = batch_loss_c / config.iter_size
        model_loss = (batch_loss_l + batch_loss_c) / config.iter_size
        batch_loss = batch_loss / config.iter_size

        loc_loss += batch_loss_l.item()
        conf_loss += batch_loss_c.item()

        ###########################
        # Logging
        ###########################

        if is_on_first_rank(config):
            config.tb.add_scalar("train/loss_l", batch_loss_l.item(), iteration)
            config.tb.add_scalar("train/loss_c", batch_loss_c.item(), iteration)
            config.tb.add_scalar("train/loss", batch_loss.item(), iteration)

        if iteration % config.print_freq == 0:
            t_finish = time.time()
            t_elapsed = t_finish - t_start
            t_start = time.time()
            logger.info('{}: iter {} epoch {} || Loss: {:.4} || Time {:.4}s || lr: {} || KL loss: {}'.format(
                config.rank, iteration, epoch, model_loss.item(), t_elapsed, optimizer.param_groups[0]['lr'],
                kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            ))

    if config.metrics_dump is not None:
        write_metrics(best_mAp, config.metrics_dump)


if __name__ == '__main__':
    main(sys.argv[1:])
