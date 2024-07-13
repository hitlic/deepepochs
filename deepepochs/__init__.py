"""
@author: liuchen
DeepEpochs is a simple Pytorch deep learning model training tool(see https://github.com/hitlic/deepepochs).
"""

__version__ = '0.6.4'

from .loops import *
from .trainer import Trainer, TrainerBase, EpochTask
from .tools import *
from .callbacks import *
from .optimizer import Optimizer, Optimizers
from .patches import PatchBase, ValuePatch, TensorPatch, MeanPatch, ConfusionPatch, ConfusionMetrics
