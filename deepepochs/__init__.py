"""
@author: hitlic
TODO:
    混合精度
    分布式
"""
__version__ = '0.4.17'

from .loops import *
from .trainer import Trainer, TrainerBase, EpochTask
from .tools import *
from .callbacks import *
from .optimizer import Optimizer, Optimizers
from .patches import PatchBase, ValuePatch, TensorPatch, MeanPatch, ConfusionPatch
