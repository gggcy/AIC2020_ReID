# -*- coding: UTF-8 -*- 
from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from torch.autograd import Variable

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
import numpy as np
import pickle


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def attribute_extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _, _, _, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def attribute_extract_features_simulation(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _, _,) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def attribute_pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _, _, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def attribute_pairwise_distance_simulation(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _, in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _, in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist



def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    #cmc_configs = {
    #    'allshots': dict(separate_camera_set=False,
    #                     single_gallery_shot=False,
    #                     first_match_break=False),
    #    'cuhk03': dict(separate_camera_set=True,
    #                   single_gallery_shot=True,
    #                   first_match_break=False),
    #    'market1501': dict(separate_camera_set=False,
    #                       single_gallery_shot=False,
    #                       first_match_break=True)}
    #cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
    #                        query_cams, gallery_cams, **params)
    #              for name, params in cmc_configs.items()}
    
    #print('CMC Scores{:>12}{:>12}{:>12}'
    #      .format('allshots', 'cuhk03', 'market1501'))
    #for k in cmc_topk:
    #    print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
    #          .format(k, cmc_scores['allshots'][k - 1],
    #                  cmc_scores['cuhk03'][k - 1],
    #                  cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return mAP


def attribute_evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _, _, _, _, _ in query]
        gallery_ids = [pid for _, pid, _, _, _, _, _, _ in gallery]
        query_cams = [cam for _, _, cam, _, _, _, _, _ in query]
        gallery_cams = [cam for _, _, cam, _, _, _, _, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return mAP

def attribute_evaluate_all_s(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _, _ in query]
        gallery_ids = [pid for _, pid, _, _, _ in gallery]
        query_cams = [cam for _, _, cam, _, _ in query]
        gallery_cams = [cam for _, _, cam, _, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, is_attribute=False, rerank=False):
        if is_attribute:
            features, _ = attribute_extract_features(self.model, data_loader)
            distmat = attribute_pairwise_distance(features, query, gallery, metric=metric)
            return attribute_evaluate_all(distmat, query=query, gallery=gallery)
        else:
            features, _ = extract_features(self.model, data_loader)
            distmat = pairwise_distance(features, query, gallery, metric=metric)
            mAP = evaluate_all(distmat, query=query, gallery=gallery)
            return mAP 

class Evaluator_simulation(object):
    def __init__(self, model):
        super(Evaluator_simulation, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, is_attribute=False, rerank=False):
        if is_attribute:
            features, _ = attribute_extract_features_simulation(self.model, data_loader)
            distmat = attribute_pairwise_distance_simulation(features, query, gallery, metric=metric)
            return attribute_evaluate_all_s(distmat, query=query, gallery=gallery)
        else:
            features, _ = extract_features(self.model, data_loader)
            distmat = pairwise_distance(features, query, gallery, metric=metric)
            mAP = evaluate_all(distmat, query=query, gallery=gallery)
            return mAP 



class Evaluator_pkl(object):
    def __init__(self, g_pkl):
        super(Evaluator_pkl, self).__init__()
        self.g_pkl = g_pkl

    def evaluate(self, data_loader, query, gallery, metric=None, is_attribute=False, rerank=False):
        if is_attribute:
            features, _ = attribute_extract_features(self.model, data_loader)
            distmat = attribute_pairwise_distance(features, query, gallery, metric=metric)
            return attribute_evaluate_all(distmat, query=query, gallery=gallery)
        else:
            features= extract_features_pkl(self.g_pkl)
            distmat = pairwise_distance_pkl(features, query, gallery, metric=metric)
            mAP = evaluate_all(distmat, query=query, gallery=gallery)
            return mAP 


def extract_features_pkl(pkl_dir):

    f = open(pkl_dir) #二进制格式读文件
    img2deepft_dict = pickle.load(f)
    features = OrderedDict()
    for fname, output in img2deepft_dict.items():
        features[fname] = Variable(torch.from_numpy(output))

    return features

def pairwise_distance_pkl(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


class Evaluator_dismat(object):
    def __init__(self, dismat_dir):
        super(Evaluator_dismat, self).__init__()
        self.dismat_dir = dismat_dir

    def evaluate(self, data_loader, query, gallery, metric=None, is_attribute=False, rerank=False):
        if is_attribute:
            features, _ = attribute_extract_features(self.model, data_loader)
            distmat = attribute_pairwise_distance(features, query, gallery, metric=metric)
            return attribute_evaluate_all(distmat, query=query, gallery=gallery)
        else:
            # features= extract_features_pkl(self.g_pkl)
            distmat = load_pickle(self.dismat_dir)
            mAP = evaluate_all(distmat, query=query, gallery=gallery)
            return mAP 


def load_pickle(pickle_file):
    f = open(pickle_file, 'rb')
    features = pickle.load(f)
    f.close()
    return features
