# -*- coding: utf-8 -*
import cPickle as pickle
import numpy as np
from sys import version_info

f1 = open('./val_pkls/rerank_densenet_0408.pkl')
f2 = open('./val_pkls/rerank_mgnnew_0408.pkl')
f3 = open('./val_pkls/rerank_152SAC_0408.pkl')
f4 = open('./val_pkls/rerank_senet152_0408.pkl')
inf1 = pickle.load(f1)
inf2 = pickle.load(f2)
inf3 = pickle.load(f3)
inf4 = pickle.load(f4)
distmat = inf1 + inf2*0.75 + inf3*0.5 + inf4
#import pdb;pdb.set_trace()
with open('./cat_distmat/cat_4_most.pkl', 'wb') as w:
    pickle.dump(distmat, w)
                              
