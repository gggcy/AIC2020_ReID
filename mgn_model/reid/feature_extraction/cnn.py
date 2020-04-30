from __future__ import absolute_import
from collections import OrderedDict
import torch
from torch.autograd import Variable

from ..utils import to_torch

def extract_cnn_feature(model, inputs):
    model.eval()
    with torch.no_grad():
        inputs = Variable(inputs).cuda()
    outputs = model(inputs)
    final_fea = outputs[-1]
    return final_fea.data.cpu()


