"""
@author: liuchen
"""
from typing import List, Dict, Union
import torch
from ..patches import PatchBase
from .trainer_base import TrainerBase


# class Trainer(TrainerBase):
#     def train_step(self,
#                    batch_x: Union[torch.Tensor, List[torch.Tensor]],
#                    batch_y: Union[torch.Tensor, List[torch.Tensor]],
#                    **step_args
#                    ) -> Dict[str, PatchBase]:
#         """
#         TODO: 非常规训练可重写本方法
#         Args:
#             batch_x:    一个mini-batch的模型输入
#             batch_y:    一个mini-batch的标签或targets
#             step_args:  当使用EpochTask时，EpochTask的step_args参数
#         Returns:
#             None 
#               或
#             dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
#         """
#         model_out = self.model(*batch_x)
#         # self.loss是对Trainer中loss参数的封装，在训练中会自动调用opt.zero_grad、loss.backward、opt.step等方法
#         self.loss(model_out, batch_y)

#     def evaluate_step(self,
#                       batch_x: Union[torch.Tensor, List[torch.Tensor]],
#                       batch_y: Union[torch.Tensor, List[torch.Tensor]],
#                       **step_args
#                       ) -> Dict[str, PatchBase]:
#         """
#         TODO: 非常规验证或测试可重写本方法，或定义val_step方法、test_step方法
#         Args:
#             batch_x:    一个mini-batch的模型输入
#             batch_y:    一个mini-batch的标签或targets
#             step_args:  当使用EpochTask时，EpochTask的step_args参数
#         Returns:
#             None 
#               或
#             dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
#         """
#         # self.model是对Trainer中model参数的封装，
#         model_out = self.model(*batch_x)
#         # self.loss是对Trainer中loss参数的封装，在训练中会自动调用opt.zero_grad、loss.backward、opt.step等方法
#         self.loss(model_out, batch_y)
