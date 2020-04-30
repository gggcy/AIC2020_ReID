from __future__ import absolute_import

from .mgn_loss import MGN_loss
from .xentropy_sac import XentropyLoss_SAC
from .triplet import TripletLoss

__all__ = ['MGN_loss','XentropyLoss_SAC','TripletLoss']
