from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import pdb

import math
import numpy as np
import sys
import torch
vreid_dir = os.getcwd()
sys.path.append(vreid_dir)
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.loss import XentropyLoss_SAC, MGN_loss, TripletLoss

from reid import datasets
from reid import models
from reid.models import ResNet_reid_50
from reid.lr_scheduler import WarmupMultiStepLR
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer, Cross_Trihard_Trainer, Trainer_SAC_Triplet
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, big_height, big_width, target_height, target_width, batch_size, num_instances,
        workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id, download=False)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.ResizeRandomCrop(big_height, big_width, target_height, target_width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(target_height, target_width),
        T.ToTensor(),
        normalizer,
    ])
    
    sampler =  RandomIdentitySampler(train_set, num_instances)
    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=sampler,
        pin_memory=True, drop_last=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print(args)
    # Create data loaders
    if args.big_height is None or args.big_width is None or args.target_height is None or args.target_width is None:
        args.big_height, args.big_width, args.target_height, args.target_width = (256, 256, 224, 224)
    dataset, num_classes, train_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.big_height, args.big_width,
                 args.target_height, args.target_width, args.batch_size, args.num_instances,
                 args.workers, args.combine_trainval)

    # Create models
    model = models.create(name=args.arch, num_classes=num_classes, num_features=args.features, norm=True)
    print(model)
    # Load from checkpoint
    start_epoch = best = 0

    if args.weights and hasattr(model,'base'): 
        print('loading resnet50')
        checkpoint = load_checkpoint(args.weights)
        del(checkpoint['fc.weight'])
        del(checkpoint['fc.bias'])
        model.base.load_state_dict(checkpoint)
    if args.resume and not args.weights:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best))
    
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    if args.arch == 'ResNet50_mgn_lr' or args.arch == 'ResNet101_mgn_lr' or args.arch == 'ResNet152_mgn_lr' :
        criterion = MGN_loss(margin1=1.2, num_instances=4, alpha=1.0, gamma =1.0,theta=0.1, 
                         has_trip = True).cuda()
    elif args.arch == 'ResNet_reid_50' or args.arch == 'ResNet_reid_101' or args.arch == 'ResNet_reid_152':
        # criterion = XentropyLoss_SAC(theta=0.2,gamma=1).cuda()
        ranking_loss = nn.MarginRankingLoss(margin=args.margin).cuda()
        criterion = { 'XentropyLoss_SAC': XentropyLoss_SAC(theta=0.2,gamma=1).cuda(), \
                    'trihard': TripletLoss(ranking_loss).cuda() }
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer
    frozen_layerName = ['conv1', 'bn1', 'relu','maxpool', 'layer1', 'layer2',]
    ##### Optimizer
    if args.frozen_sublayer:
        frozen_Source = None
        if hasattr(model.module,'base'):
            frozen_Source = 'model.module.base.'
        elif hasattr(model.module,frozen_layerName[0]):
            frozen_Source = 'model.module.'
        else:
            raise RuntimeError('Not freeze layers but frozen_sublayer is True!')

        base_params_set = set()
        for subLayer in frozen_layerName:
            if hasattr( eval(frozen_Source[:-1]),subLayer):
                print('frozen layer: ', subLayer )
                single_module_param = eval(frozen_Source + subLayer + '.parameters()')
                # base_params.append(single_module_param)
                single_module_param_set = set(map(id, single_module_param))
                base_params_set = base_params_set | single_module_param_set
            else:
                print("current model doesn't have ",subLayer)

        new_params = [p for p in model.parameters() if
                      id(p) not in base_params_set]

        base_params = [p for p in model.parameters() if
                       id(p) in base_params_set]
        param_groups = [
            {'params': base_params, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}
        ]
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    if args.arch == 'ResNet50_mgn_lr' or args.arch == 'ResNet101_mgn_lr' or args.arch == 'ResNet152_mgn_lr' :
        trainer = Trainer(model,criterion)

    elif args.arch == 'ResNet_reid_50' or args.arch == 'ResNet_reid_101' or args.arch == 'ResNet_reid_152':

        trainer = Trainer_SAC_Triplet(model, criterion, metric_loss_weight=args.metric_loss_weight)
    else:
        trainer = Cross_Trihard_Trainer(model, criterion, metric_loss_weight=args.metric_loss_weight)

    # Schedule learning rate
    print(args.step_epoch)
    scheduler = WarmupMultiStepLR(optimizer, args.step_epoch, gamma = args.gamma, warmup_factor=args.warm_up_factor, warmup_iters = args.warm_up_iter)
    # Start training
    for epoch in range(start_epoch+1, args.epochs+1):
        scheduler.step()
        trainer.train(epoch, train_loader, optimizer)
        
        if epoch % args.epoch_inter == 0 or epoch >=args.dense_evaluate:
            tmp_mAP, tmp_res = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
            if epoch >= args.start_save:
                if tmp_mAP > best :
                    best = tmp_mAP
                    flag = True
                else:
                    flag = False
                # save_checkpoint({
                #     'state_dict': model.module.state_dict(),
                #     'epoch': epoch,
                #     'best_map':tmp_mAP
                # }, flag, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                    'best_map':tmp_mAP
                }, flag, fpath=osp.join(args.logs_dir, 'pass%d.pth.tar' % (epoch)))
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ResNet_reid_50")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--big_height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--big_width', type=int, default=256,
                        help="input width, default: 256")
    parser.add_argument('--target_height', type=int, default=224,
                        help="crop height, default: 224")
    parser.add_argument('--target_width', type=int, default=224,
                        help="crop width, default: 224")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='ResNet_reid_50')
    parser.add_argument('--frozen_sublayer', type=lambda x: (str(x).lower() == 'true'),default=True)
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--lr_mult', type=float, default=1.0)
    parser.add_argument('--metric_loss_weight', type=float, default=0.5)
    parser.add_argument('--gamma',type=float,default = 0.1)
    parser.add_argument('--warm_up_factor',type=float, default=0.01)
    parser.add_argument('--warm_up_iter', type=int, default=100)
    parser.add_argument('--step_epoch', nargs='+', type=int)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--weights', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--epoch_inter', type=int, default=10)
    parser.add_argument('--dense_evaluate', type=int, default=90)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
