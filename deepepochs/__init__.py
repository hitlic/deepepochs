"""
@author: hitlic
TODO:
    混合精度
    分布式
"""
__version__ = '0.4.11'

from .loops import *
from .trainer import Trainer, TrainerBase, EpochTask
from .utils import *
from .callbacks import *
