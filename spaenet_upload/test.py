from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from reid import datasets
from reid import models
from reid.models.o2cap import CameraAwareMemory
from reid import trainers
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import ClassUniformlySampler
from reid.utils.data.preprocessor import Preprocessor, CameraAwarePreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.faiss_rerank import compute_jaccard_distance
from bisect import bisect_right


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print('root path= {}'.format(root))
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = ClassUniformlySampler(train_set, class_position=4, k=num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(CameraAwarePreprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def create_model(args):
    if args.arch=='resnet50':
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pool_type=args.pool_type)
    elif args.arch=='resnet50_ibn':
        #model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pool_type=args.pool_type)
        model = models.create(args.arch, num_classes=0,
                              pool_type=args.pool_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main(args):
    #args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = False

    sys.stdout = Logger(osp.join(args.logs_dir, 'test_log_5_24.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)

    # create image-level camera information
    all_img_cams = torch.tensor([c for _, _, c in sorted(dataset.train)])
    temp_all_cams = all_img_cams.numpy()
    all_img_cams = all_img_cams.cuda()
    unique_cameras = torch.unique(all_img_cams)

    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    model.load_state_dict(torch.load("44net_params.pth"))
    # Create memory
    memory = CameraAwareMemory(temp=args.temp, momentum=args.momentum, all_img_cams=all_img_cams,
                               has_cross_cam_loss=True, has_online_proxy_loss=True, posK=3).cuda()

    
    # Evaluator
    evaluator = Evaluator(model)


    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)


    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAP enhancement for unsupervised re-ID")
    # data
    parser.add_argument('--dataset', type=str, default='DukeMTMC-reID',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=20,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_ibn')
    parser.add_argument('-pool_type', type=str, default='avgpool')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.07,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='/media/npu-tao/tao/DQY/dataset/')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='test_logs/')
    args = parser.parse_args()    

    main(args)
