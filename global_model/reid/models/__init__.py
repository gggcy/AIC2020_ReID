from __future__ import absolute_import

from .resnet import *
from .multi_attribute_3 import *
from .cross_entropy_trihard import *
from .cross_trihard_senet import *
from .cross_trihard_se_resnet import *
from .direction import *
from .multi_attribute_8 import *
from .multi_attribute_8_152 import *
from .multi_attribute_2_152_s import *
from .hrnet_48w import HighResolutionNet_reid48w
from .se_152_ibn import *
from .dense_ibn_a import *
from .dpn import *
from .densenet import *
from .senet154 import *
from .res2net import *
from .inceptionv4 import *
__factory = {
    'multi_attribute_3_resnet50':multi_attribute_3_resnet50,
    'cross_entropy_trihard_resnet101':cross_entropy_trihard_resnet101,
    'cross_entropy_trihard_resnet152':cross_entropy_trihard_resnet152,
    'cross_trihard_senet101':cross_trihard_senet101,
    'cross_trihard_se_resnet152':cross_trihard_se_resnet152,
    'direction_resnet50':direction_resnet50,
    'multi_attribute_8_resnet50':multi_attribute_8_resnet50,
    'multi_attribute_8_resnet152':multi_attribute_8_resnet152,
    'multi_attribute_2_resnet152_s':multi_attribute_2_resnet152_s,
    'Hrnet48':HighResolutionNet_reid48w,
    'se_152_ibn':se_152_ibn,
    'densenet169_ibn_a':densenet169_ibn_a,
    'dpn107':dpn107,
    'densenet161':densenet161,
    'senet154':senet154,
    'res2net101':res2net101_v1b,
    'inceptionv4': inceptionv4,

}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
