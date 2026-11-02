"""
@author: liuchen
DeepEpochs is a simple Pytorch deep learning model training tool(see https://github.com/hitlic/deepepochs).
"""

__version__ = '0.7.3'

from .callbacks import *
from .loops import *
from .launcher import launch
from .optimizer import Optimizer, Optimizers
from .patches import ConfusionMetrics, ConfusionPatch, MeanPatch, PatchBase, TensorPatch, ValuePatch
from .tools import *
from .trainer import EpochTask, Trainer, TrainerBase
