from __future__ import absolute_import
import warnings

from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .viper import VIPeR
from .small_vehicle import Small_Vehicle
from .aicity_car196 import Aicity_Car196
from .complete_aicity_car import Complete_Aicity_Car
from .gao_crop_train import Gao_Crop_Train
from .new_complete_aicity_car import New_Complete_Aicity_Car
from .aicity_attribute import Aicity_Attribute
from .new_train import New_Train


__factory = {
    'viper': VIPeR,
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'small_vehicle': Small_Vehicle,
    'aicity_car196': Aicity_Car196,
    'complete_aicity_car':Complete_Aicity_Car,
    'gao_crop_train':Gao_Crop_Train,
    'new_complete_aicity_car':New_Complete_Aicity_Car,
    'aicity_attribute': Aicity_Attribute,
    'new_train':New_Train,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
