# encoding=utf8
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
root_dir=os.getcwd()
import sys
sys.path.append(root_dir)
import math
import time
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
import pickle
from reid.dist_metric import DistanceMetric
import models
from reid.utils_p1.data import transforms as T
from reid.utils_p1.meters import AverageMeter
from reid.utils_p1.serialization import load_checkpoint, save_checkpoint
from reid.utils_p1.data import Flip_Preprocessor
from reid.feature_extraction import extract_cnn_feature
import torchvision.transforms
import pdb
import sys

class Combine_Net(torch.nn.Module):
    def __init__(self, arch='cross_entropy_trihard_resnet50', num_classes=529, num_features=1024):
        super(Combine_Net, self).__init__()
        self.net = models.create(arch, num_classes=num_classes, num_features=num_features)
    def forward(self, x):
        out = self.net(x) 
        return out


def extract_features(model, data_loader, is_flip=False, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()

    end = time.time()
    if is_flip:
        print('flip')
        for i, (imgs, flip_imgs, fnames) in enumerate(data_loader):
            data_time.update(time.time() - end)
    
            outputs = extract_cnn_feature(model, imgs)
            flip_outputs = extract_cnn_feature(model, flip_imgs)
            final_outputs = (outputs + flip_outputs) / 2 
            for fname, output in zip(fnames, final_outputs):
                features[fname] = output.numpy()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
    else:
        print('no flip')
        for i, (imgs, fnames) in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            outputs = extract_cnn_feature(model, imgs)
            for fname, output in zip(fnames, outputs):
                features[fname] = output.numpy()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg)) 
    return features


def get_real_test_data(query_dir, gallery_dir, target_height, target_width, batch_size, workers):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(target_height, target_width),
        torchvision.transforms.CenterCrop((288, 384)),

        T.ToTensor(),
        normalizer,
    ])  
    
    query_loader = DataLoader(
        Flip_Preprocessor(data_dir=query_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False,pin_memory=True)
    
    gallery_loader = DataLoader(
        Flip_Preprocessor(data_dir=gallery_dir, 
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False,pin_memory=True)

    return query_loader, gallery_loader


def main(args):
    print('running...')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    print(args)
    # Create data loaders
    if args.target_height is None or args.target_width is None:
        args.target_height, args.target_width = (288, 384)

    # Create model
    model = Combine_Net(arch=args.arch, num_classes=args.num_classes, num_features=args.num_features)
    print(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    model.net.load_state_dict(checkpoint['state_dict']) 
    #model = nn.DataParallel(model, [0,1]).cuda()
    model = nn.DataParallel(model, [0,1,2,3]).cuda()
    query_loader, gallery_loader = get_real_test_data(args.query_dir, args.gallery_dir, args.target_height, args.target_width, args.batch_size, args.workers)
    query_features = extract_features(model, query_loader, is_flip=True)
    gallery_features = extract_features(model, gallery_loader, is_flip=True)    
    q_file = open(args.q_file, 'wb')
    g_file = open(args.g_file, 'wb')
    pickle.dump(query_features, q_file)
    pickle.dump(gallery_features, g_file)
    q_file.close()
    g_file.close()
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract features")
    # data
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default='/data/home/cunyuangao/Project/AIC2020_ReID/mgn/logs/resnet_reid_152/model_best.pth.tar')
    parser.add_argument('--target_height', type=int, default=310,
                        help="crop height, default: 224")
    parser.add_argument('--target_width', type=int, default=414,
                        help="crop width, default: 224")
    #parser.add_argument('--query_dir', type=str, default='./examples/data/image_query')
    parser.add_argument('--query_dir', type=str, default='/data/home/cunyuangao/AIC20_ReID/image_query')
    parser.add_argument('--gallery_dir', type=str, default='/data/home/cunyuangao/AIC20_ReID/image_test')
    # model
    parser.add_argument('--arch', type=str, default='HighResolutionNet_reid') 
    parser.add_argument('--num_classes', type=int, default=1645)
    parser.add_argument('--num_features', type=int, default=2048) 
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--q_file', type=str, default='query_se_res152.pkl')
    parser.add_argument('--g_file', type=str, default='gallery_se_res152.pkl')
    # misc
    working_dir = os.getcwd()
    dataset_dir = os.path.join(os.path.dirname(working_dir),'dataset')
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default= dataset_dir)
    main(parser.parse_args())
