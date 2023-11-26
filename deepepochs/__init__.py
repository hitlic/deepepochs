"""
@author: hitlic
TODO:
    1. Accelerate下训练指标的输出，正确性调试；
    2. Accelerate保存与加载Checkpoint
    3. Accelerate下Earlystop的实现
    4. Accelerate下利用tensorboard日志记录指标
    5. 在Notebook中启动Accelerate程序
    6. 混合精度训练测试
"""
__version__ = '0.5.0'

from .loops import *
from .trainer import Trainer, TrainerBase, EpochTask
from .tools import *
from .callbacks import *
from .optimizer import Optimizer, Optimizers
from .patches import PatchBase, ValuePatch, TensorPatch, MeanPatch, ConfusionPatch
