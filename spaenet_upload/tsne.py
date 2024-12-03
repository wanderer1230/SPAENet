from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.models.o2cap import CameraAwareMemory
from reid.evaluators import Evaluator, extract_features
from reid.models.resnet_ibn_a import remove_module_key
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor, CameraAwarePreprocessor
from reid.utils.logging import Logger

start_epoch = best_mAP = 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print('root path= {}'.format(root))
    dataset = datasets.create(name, root)
    return dataset




def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor()
        #normalizer
    ])

    if (testset is None):
        #testset = list(set(dataset.query) | set(dataset.gallery))
        testset = dataset.gallery
    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader



def create_model(args):
    if args.arch == 'resnet50':
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0,
                              pool_type=args.pool_type)
    elif args.arch == 'resnet50_ibn':
        # model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pool_type=args.pool_type)
        model = models.create(args.arch, num_classes=0,
                              pool_type=args.pool_type)
    # use CUDA
    model.cuda()
    if args.visual != True:
        model = nn.DataParallel(model)
    return model


def main(args):
    # args = parser.parse_args()

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

    sys.stdout = Logger(osp.join(args.logs_dir, 'tsne_log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)

    # create image-level camera information
    all_img_cams = torch.tensor([c for _, _, c in sorted(dataset.train)])
    temp_all_cams = all_img_cams.numpy()
    all_img_cams = all_img_cams.cuda()

    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    state_dict = torch.load("49net_params.pth", map_location=torch.device('cpu'))
    state_dict = remove_module_key(state_dict)
    model.load_state_dict(state_dict)

    #model.load_state_dict(torch.load("49net_params.pth"))

    # Evaluator
    evaluator = Evaluator(model)

    evaluator.tsne(test_loader)
    #evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
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
    parser.add_argument('--visual', type=int, default=True)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.07,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='/media/npu-tao/tao/DQY/dataset/')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='train_logs/')
    args = parser.parse_args()

    main(args)

