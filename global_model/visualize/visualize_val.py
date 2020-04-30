import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

root_path = '/data/home/cunyuangao/Project/track2.txt'
image_path = ' '
########################################################################
# Visualize the rank result
gallery_path = '/data/home/public/data/AIC2020/AIC20_track2/AIC20_ReID/image_test/'
query_path = '/data/home/public/data/AIC2020/AIC20_track2/AIC20_ReID/image_query/'
lines = open(root_path).readlines()
count = 0
for line in lines[:1]:
    count += 1 
    words = line.strip('\n').split(' ')


    print('Top 20 images are as follow:')

    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path + str.zfill(count),'query')
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path = image_path + words[i]
        # import pdb;pdb.set_trace()
        imshow(img_path)
        print(img_path)

    fig.savefig("show.png")