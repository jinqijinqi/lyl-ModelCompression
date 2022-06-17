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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.utils.data as data
from examples.my_common.model_loader import load_resuming_model_state_dict_and_checkpoint_from_path
from examples.my_common.sample_config import create_sample_config
from examples.object_detection.prunning import prunning_weights

from examples.my_common.argparser import get_common_argument_parser
from examples.my_common.distributed import DistributedSampler, configure_distributed
from examples.my_common.example_logger import logger
from examples.my_common.execution import ExecutionMode, get_device, get_execution_mode
from examples.my_common.execution import start_worker

from examples.my_common.utils import configure_paths,is_on_first_rank, configure_logging, print_args
from examples.my_common.utils import write_metrics
from examples.object_detection.my_dataset import detection_collate, get_testing_dataset, get_training_dataset
from examples.object_detection.my_eval import test_net

from examples.object_detection.my_model import build_ssd
from my_nncf.checkpoint_loading import load_state
from my_nncf.dynamic_graph.graph_builder import create_input_infos


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
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)
    if is_on_first_rank(config):
        configure_logging(logger, config)
        print_args(vars(config))

    config.device = get_device(config)
    config.start_iter = 0

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)
    ###########################
    # Prepare data
    ###########################
    test_data_loader = create_dataloaders(config)
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

    with torch.no_grad():
        with torch.no_grad():
            net.eval()
            mAp = test_net(net, config.device, test_data_loader, distributed=config.distributed)
            if config.metrics_dump is not None:
                write_metrics(mAp, config.metrics_dump)
            return


def create_dataloaders(config):
    logger.info('Loading Dataset...')
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
    return test_data_loader

def create_model(config, resuming_model_sd=None):
    input_info_list = create_input_infos(config.nncf_config)
    image_size = input_info_list[0].shape[-1]
    thresholds = [-3, -2.5, -2.5, -2.5, -2.5, -4, -4, -4, 15, 9, 4, 2,
                  -2.5, -3, -1, -2, -1, -2, -1, -3, -2, -2.25, -2]

    ssd_bayesian = build_ssd('ssd_vgg_bayesian', config.ssd_params, image_size, config.num_classes, config)
    load_state(ssd_bayesian, resuming_model_sd, is_resume=True)

    ssd_pruned = build_ssd('ssd_vgg_pruned', config.ssd_params, image_size, config.num_classes, config)
    # 获取剪枝后的权值
    pruned_weights = prunning_weights(ssd_pruned, ssd_bayesian, *ssd_bayesian.get_masks(thresholds))
    torch.save(pruned_weights,"./examples/object_detection/ckpt/ssd_pruned.pth")
    load_state(ssd_pruned, pruned_weights, is_resume=True)
    ssd_pruned.to(config.device)

    return ssd_pruned


if __name__ == '__main__':
    main(sys.argv[1:])
