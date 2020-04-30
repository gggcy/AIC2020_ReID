from __future__ import absolute_import
import os.path as osp
import os
from PIL import Image
import pickle
import torch
import torchvision.transforms as T


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


class Flip_Preprocessor(object):
    def __init__(self, data_dir=None, transform=None):
        super(Flip_Preprocessor, self).__init__()
        self.data_dir = data_dir 
        self.transform = transform
        self.img_names = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.img_names[index]
        fpath = fname
        if self.data_dir is not None:
            fpath = osp.join(self.data_dir, fname)
        img = Image.open(fpath).convert('RGB')
        flip_img = T.functional.hflip(img)
        if self.transform is not None:
            img = self.transform(img)
            flip_img = self.transform(flip_img)
        return img, flip_img, fname


class Direct_Preprocessor(object):
    def __init__(self, data_dir=None, transform=None, is_train=True):
        super(Direct_Preprocessor, self).__init__()
        self.data_dir = data_dir 
        self.transform = transform
        self.img_names = os.listdir(self.data_dir)
        self.is_train = is_train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.img_names[index]
        if self.is_train:
            direct = int(fname.split('_')[2])
        else:
            direct = int(fname.split('_')[1])   
        fpath = fname
        assert self.transform is not None
        if self.data_dir is not None:
            fpath = osp.join(self.data_dir, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, direct


class Attribute_Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Attribute_Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, color, car_type, roof, window, logo = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, color, car_type, roof, window, logo


class Attribute_Preprocessor_s(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Attribute_Preprocessor_s, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, color, car_type= self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, color, car_type



class Flip_Preprocessor_For_Vis(object):
    def __init__(self, data_dir=None, transform=None):
        super(Flip_Preprocessor_For_Vis, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = data_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.img_names[index]
        fpath = fname
        if self.data_dir is not None:
            fpath = osp.join(self.data_dir, fname)
        img = Image.open(fpath).convert('RGB')
        flip_img = T.functional.hflip(img)
        if self.transform is not None:
            img = self.transform(img)
            flip_img = self.transform(flip_img)
        return img, flip_img, fname