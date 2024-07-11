"""
@author: liuchen
"""
import math
from typing import List, Dict, Union
import torch
from ..tools import batches
from ..patches import PatchBase
from .trainer_base import TrainerBase


class Trainer(TrainerBase):
    def train_step(self,
                   batch_x: Union[torch.Tensor, List[torch.Tensor]],
                   batch_y: Union[torch.Tensor, List[torch.Tensor]],
                   **step_args
                   ) -> Dict[str, PatchBase]:
        """
        实现了累积梯度训练。
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        if self.grad_accumulate_steps == 1:
            model_out = self.model(*batch_x)
            # self.loss是对Trainer中loss参数的封装，会自动调用opt.zero_grad、loss.backward、opt.step等方法
            self.loss(model_out, batch_y)
            return

        # 累积梯度训练
        b_size = self.find_batch_size(batch_x)
        sub_batch_size = math.ceil(b_size / self.grad_accumulate_steps)
        for sub_batch_idx, (sub_batch_x, sub_batch_y) in enumerate(zip(batches(batch_x, sub_batch_size), batches(batch_y, sub_batch_size))):
            # 将子批量数据放入GPU
            sub_batch_x, sub_batch_y = sub_batch_x.to(self.device), sub_batch_y.to(self.device)
            if self.accelerator is None:
                model_out = self.model(*sub_batch_x)
                self.loss(model_out, sub_batch_y, sub_batch_idx + 1 < self.grad_accumulate_steps)
            else:
                with self.accelerator.accumulate(self.model.model):
                    model_out = self.model(*sub_batch_x)
                    self.loss(model_out, sub_batch_y, sub_batch_idx + 1 < self.grad_accumulate_steps)

    def evaluate_step(self,
                      batch_x: Union[torch.Tensor, List[torch.Tensor]],
                      batch_y: Union[torch.Tensor, List[torch.Tensor]],
                      **step_args
                      ) -> Dict[str, PatchBase]:
        """
        TODO: 非常规验证或测试可修改本方法中的代码。也可以定义val_step方法或test_step方法。
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        # self.model是对Trainer中model参数的封装，
        model_out = self.model(*batch_x)
        # self.loss是对Trainer中loss参数的封装，会自动调用opt.zero_grad、loss.backward、opt.step等方法
        self.loss(model_out, batch_y)
