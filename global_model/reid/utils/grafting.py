import numpy as np
import torch


def entropy(x,n=10):
    x=x.reshape(-1)
    scale=(x.max()-x.min())/n
    entropy=0
    for i in range(n):
        p=torch.sum((x>=x.min()+i*scale)*(x<x.min()+(i+1)*scale),dtype=torch.float)/len(x)
        if p!=0:
            entropy-=p*torch.log(p)
    return entropy


def filter_l1norm(weights):
    l1_norm = torch.norm(weights.reshape(weights.shape[0], -1), p = 1, dim=1)
    return l1_norm