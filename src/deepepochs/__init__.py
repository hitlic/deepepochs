"""
@author: liuchen
DeepEpochs is a simple Pytorch deep learning model training tool(see https://github.com/hitlic/deepepochs).
"""

__version__ = '0.7.5'

from .callbacks import *  # noqa: F401, F403
from .launcher import launch  # noqa: F401, F403
from .loops import *  # noqa: F401, F403
from .optimizer import Optimizer, Optimizers  # noqa: F401, F403
from .patches import ConfusionMetrics, ConfusionPatch, MeanPatch, PatchBase, TensorPatch, ValuePatch  # noqa: F401, F403
from .tools import *  # noqa: F401, F403
from .trainer import EpochTask, Trainer, TrainerBase  # noqa: F401, F403
