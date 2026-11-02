import os
import platform
import random
import subprocess
import sys
from os import path as osp

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.tensorboard.summary import hparams

from ..loops import _color, check_path
from .callback import Callback


class _KeywordInputModel(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)

    def forward(self, *inputs):
        return self.model(**dict(zip(self.input_names, inputs)))


class LogCallback(Callback):
    def __init__(self, log_dir='./logs', log_graph=False, example_input=None):
        """
        Args:
            log_dir:        日志保存位置
            log_graph:      是否保存模型结构图
            example_input:  保存模型结构图时的输入样例，默认以第一个训练batch_x作为样例输入
        """
        super().__init__(priority=1)
        self.log_graph = log_graph
        self.example_input = example_input
        self.graph_saved = False
        self._graph_attempted = False

        self.log_dir=os.path.abspath(log_dir)
        self.global_train_batch_idx = 0
        self.global_train_epoch_idx = 0
        self.global_val_batch_idx = 0
        self.global_val_epoch_idx = 0

    def state_dict(self):
        return {name: getattr(self, name) for name in (
            'global_train_batch_idx', 'global_train_epoch_idx',
            'global_val_batch_idx', 'global_val_epoch_idx')}

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def log(self, metrics, stage, loop_phase, idx):
        for k, v in metrics.items():
            self.logger.add_scalar(f'{k}/{loop_phase}/{stage}', float(v), idx)

    def log_hparams(self, hyper_params, metrics):
        # self.logger.add_hparams(hyper_params, metrics)
        # 下面的方式可以将超参数写入已有文件之中
        exp, ssi, sei = hparams(hyper_params, metrics)
        self.logger.file_writer.add_summary(exp)
        self.logger.file_writer.add_summary(ssi)
        self.logger.file_writer.add_summary(sei)
        for k, v in metrics.items():
            self.logger.add_scalar(k, float(v))

    def on_before_fit(self, trainer, epochs):
        if not trainer.main_process:
            return
        self._graph_attempted = False
        log_dir = osp.join(self.log_dir, trainer.running_id)
        check_path(log_dir)
        logger = getattr(trainer, 'logger', None)
        if logger is None:
            self.logger = SummaryWriter(log_dir=log_dir)
            trainer.logger = self.logger
        else:
            self.logger = logger

    def on_before_test_epochs(self, trainer, tasks):
        # test 可以直接用于已载入的模型，不要求先调用 fit 创建 writer。
        if trainer.main_process and getattr(self, 'logger', None) is None:
            self.on_before_fit(trainer, trainer.max_epochs)

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        """保存模型结构图"""
        if not trainer.main_process:
            return
        if self.log_graph and not self.graph_saved and not self._graph_attempted:
            self._graph_attempted = True
            model_input = batch_x if self.example_input is None else self.example_input
            try:
                graph_model = trainer.distributed.unwrap_serialization_model(trainer.model.model)
                training_modes = [(module, module.training) for module in graph_model.modules()]
                if isinstance(model_input, dict):
                    graph_model = _KeywordInputModel(graph_model, model_input)
                    model_input = tuple(model_input.values())
                python_rng = random.getstate()
                numpy_rng = np.random.get_state()
                devices = [trainer.device.index] if trainer.device.type == 'cuda' else []
                try:
                    # 绘图会额外执行前向，不能推进训练使用的随机流。
                    with torch.random.fork_rng(devices=devices):
                        self.logger.add_graph(graph_model, model_input, use_strict_trace=False)
                finally:
                    random.setstate(python_rng)
                    np.random.set_state(numpy_rng)
                    for module, training in training_modes:
                        module.training = training
                self.graph_saved = True
            except Exception as e:
                trainer.print(_color("[WARN]", "warning"), "模型结构图保存失败:", e)

        # 记录学习率
        self.log(self.lr_dict(trainer), 'train', 'batch', self.global_train_batch_idx)

    def lr_dict(self, trainer):
        current_lr = trainer.opt.get_current_lr()
        if isinstance(current_lr, (list, tuple)):
            _lr_dict = {f'learning_rate_{i}': lr  for i, lr in enumerate(current_lr)}
        else:
            _lr_dict = {'learning_rate': current_lr}
        return _lr_dict

    def on_before_train_epoch(self, trainer, task):
        if trainer.main_process:
            self.log(self.lr_dict(trainer), 'train', 'epoch', self.global_train_epoch_idx)

    def on_after_test_epoch(self, trainer, task, metrics):
        if trainer.hyper_params is not None and trainer.main_process:
            self.log_hparams(trainer.hyper_params, metrics)

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.global_train_batch_idx += 1
        if trainer.main_process:
            self.log(metrics, 'train', 'batch', self.global_train_batch_idx)

    def on_after_train_epoch(self, trainer, task, metrics):
        self.global_train_epoch_idx = trainer.epoch_idx + 1
        if trainer.main_process:
            self.log(metrics, 'train', 'epoch', self.global_train_epoch_idx)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.global_val_batch_idx += 1
        if trainer.main_process:
            self.log(metrics, 'val', 'batch', self.global_val_batch_idx)

    def on_after_val_epoch(self, trainer, task, metrics):
        self.global_val_epoch_idx = trainer.epoch_idx + 1
        if trainer.main_process:
            self.log(metrics, 'val', 'epoch', self.global_val_epoch_idx)

    def run_tensorboard(self):
        run_tensorboard(self.log_dir)


def run_tensorboard(log_dir):
    if 'Windows' in platform.system():
        tensorboard = osp.join(sys.prefix, 'Scripts', 'tensorboard')
    else:
        tensorboard = osp.join(sys.prefix, 'bin', 'tensorboard')
    try:
        subprocess.run([tensorboard, '--logdir', os.fspath(log_dir)], check=False)
    except (OSError, KeyboardInterrupt):
        pass
