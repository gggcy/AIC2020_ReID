from __future__ import absolute_import


from .resnet_reid import ResNet_reid_50
from .resnet_reid import ResNet_reid_101
from .resnet_reid import ResNet_reid_152
from .resnet_mgn import ResNet50_mgn_lr
from .resnet_mgn import ResNet101_mgn_lr
from .resnet_mgn import ResNet152_mgn_lr
from .hrnet import HighResolutionNet_reid
from .hrnet_48w import HighResolutionNet_reid48w

__factory = {
    'ResNet_reid_50': ResNet_reid_50,
    'ResNet_reid_101': ResNet_reid_101,
    'ResNet_reid_152': ResNet_reid_152,

    'ResNet50_mgn_lr': ResNet50_mgn_lr,
    'ResNet101_mgn_lr': ResNet101_mgn_lr,
    'ResNet152_mgn_lr': ResNet152_mgn_lr,
    'HighResolutionNet_reid': HighResolutionNet_reid,
    'Hrnet48': HighResolutionNet_reid48w,
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
