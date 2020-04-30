from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable
import torch
from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, requires_grad=False)
    with torch.no_grad():
        if modules is None:
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs == outputs[-1]
            outputs = outputs.data.cpu()
            return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())


def extract_extra_attrib_feature(model, inputs, modules=None):
    model.eval()
    imgs, logo_feas, attrib_feas = inputs
    imgs, logo_feas, attrib_feas = to_torch(imgs), to_torch(logo_feas), to_torch(attrib_feas)
    imgs = Variable(imgs, requires_grad=False)
    logo_feas = Variable(logo_feas, requires_grad=False)
    attrib_feas = Variable(attrib_feas, requires_grad=False)
    if modules is None:
        outputs = model(imgs, logo_feas, attrib_feas)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
