from functools import partial

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from .callback import Callback


class GradClipCallback(Callback):
    def __init__(self, max_value, norm_type=None):
        """梯度截断。
        Args:
            max_value (float):       最大取值
            norm_type (None, float): 取值为None或正数p-norm，None表示直接对取值进行剪裁，否则通过向量的指定类型范数进行剪裁
        """
        assert norm_type is None or norm_type > 0, 'The value of norm_type is None or a positive integer.'
        self.max_value = max_value
        self.norm_type = norm_type
        self.clip = None
        super().__init__()

    def prepare_clip(self, trainer):
        if self.clip is not None:
            return

        if self.norm_type is None:
            self.clip = clip_grad_value_
        else:
            clip_fn = clip_grad_norm_
            self.clip = partial(clip_fn, norm_type=self.norm_type)

    def _parameters_to_clip(self, trainer, optimize_index):
        parameters = trainer.loss.optimizing_parameters
        optimizers = list(trainer.opt) if optimize_index is None else [trainer.opt[optimize_index]]
        if self.norm_type is None or len(optimizers) == 1:
            return parameters

        # AMP 已按优化器记录溢出。非有限组不能参与联合范数，
        # 否则其 NaN 会经同一个裁剪系数污染本来可以正常更新的组。
        # 有限组仍联合裁剪，共享参数仅计入一次，保留正常路径的范数语义。
        parameters, seen = [], set()
        for optimizer in optimizers:
            group = [p for item in optimizer.param_groups for p in item['params'] if p.grad is not None]
            if not group:
                continue
            # LossWrapper已反缩放并缓存本次AMP溢出；FP32没有该记录，需检查梯度。
            if trainer.scaler.is_enabled():
                if any(value.item() for value in trainer.scaler._found_inf_per_device(optimizer.opt).values()):
                    continue
            elif not torch.stack([torch.isfinite(p.grad).all() for p in group]).all():
                continue
            for parameter in group:
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return parameters

    def on_before_optimize(self, trainer, optimize_index):
        self.prepare_clip(trainer)
        # LossWrapper 已完成反缩放，此时梯度尚未用于参数更新。
        parameters = self._parameters_to_clip(trainer, optimize_index)
        if parameters:
            self.clip(parameters, self.max_value)
