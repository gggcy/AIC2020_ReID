from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
import pickle
import os
import numpy as np



def extract_real_test_features(model, query_loader, gallery_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    query_attribute_features = OrderedDict()
    gallery_attribute_features = OrderedDict()

    end = time.time()
    root_path = os.getcwd()
    query_attribute_feature_pickle_file = open(root_path + '/reid/pickle_file/your_query_attribute_name.pkl', 'wb')
    gallery_attribute_feature_pickle_file = open(root_path + '/reid/pickle_file/your_gallery_attribute_name.pkl', 'wb')
    
    for i, (imgs, fnames) in enumerate(query_loader):
        data_time.update(time.time() - end)

        #color, car_type, roof, window, logo = extract_cnn_feature(model, imgs)
        car_type = extract_cnn_feature(model, imgs)
        #for fname, c, t, r, w, l in zip(fnames, color, car_type, roof, window, logo):
        #    c, t, r, w, l = c.data.cpu().numpy(), t.data.cpu().numpy(), r.data.cpu().numpy(), w.data.cpu().numpy(), l.data.cpu().numpy()
        #    query_attribute_features[fname] = [c, t, r, w, l]

	for fname, t in zip(fnames, car_type):
            t = t.data.cpu().numpy()
            query_attribute_features[fname] = t 

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(query_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    pickle.dump(query_attribute_features, query_attribute_feature_pickle_file)
    query_attribute_feature_pickle_file.close()

    for i, (imgs, fnames) in enumerate(gallery_loader):
        data_time.update(time.time() - end)

        #color, car_type, roof, window, logo = extract_cnn_feature(model, imgs)
        car_type = extract_cnn_feature(model, imgs)
        #for fname, c, t, r, w, l in zip(fnames, color, car_type, roof, window, logo):
        #    c, t, r, w, l = c.data.cpu().numpy(), t.data.cpu().numpy(), r.data.cpu().numpy(), w.data.cpu().numpy(), l.data.cpu().numpy()
        #    gallery_attribute_features[fname] = [c, t, r, w, l]
	
	for fname, t in zip(fnames, car_type):
            t = t.data.cpu().numpy()
            gallery_attribute_features[fname] = t

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(gallery_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    pickle.dump(gallery_attribute_features, gallery_attribute_feature_pickle_file)
    gallery_attribute_feature_pickle_file.close()


def load_features(pickle_file_path):
    pickle_file = open(pickle_file_path, 'rb')
    features = pickle.load(pickle_file)
    pickle_file.close()
    return features

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, metric=None):
        root_path = os.getcwd()
        extract_real_test_features(self.model, query_loader, gallery_loader)
