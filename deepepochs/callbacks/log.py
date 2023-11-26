from .callback import Callback
from ..loops import check_path
import sys
import os
from os import path as osp
import platform
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


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

        self.log_dir=os.path.abspath(log_dir)
        self.global_train_batch_idx = 0
        self.global_train_epoch_idx = 0
        self.global_val_batch_idx = 0
        self.global_val_epoch_idx = 0

    def log(self, metrics, stage, loop_phase, idx):
        for k, v in metrics.items():
            self.logger.add_scalar(f'{k}/{loop_phase}/{stage}', v, idx)

    def log_hparams(self, hyper_params, metrics):
        # self.logger.add_hparams(hyper_params, metrics)
        # 下面的方式可以将超参数写入已有文件之中
        exp, ssi, sei = hparams(hyper_params, metrics)
        self.logger.file_writer.add_summary(exp)
        self.logger.file_writer.add_summary(ssi)
        self.logger.file_writer.add_summary(sei)
        for k, v in metrics.items():
            self.logger.add_scalar(k, v)

    def on_before_fit(self, trainer, epochs):
        if not trainer.main_process:
            return
        log_dir = osp.join(self.log_dir, trainer.running_id)
        check_path(log_dir)
        logger = getattr(trainer, 'logger', None)
        if logger is None:
            self.logger = SummaryWriter(log_dir=log_dir)
            trainer.logger = self.logger
        else:
            self.logger = logger

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        """保存模型结构图"""
        if not trainer.main_process:
            return
        if self.log_graph and not self.graph_saved:
            model_input = batch_x if self.example_input is None else self.example_input
            self.graph_saved = True
            try:
                self.logger.add_graph(trainer.model.model, model_input)
            except Exception as e:
                print("模型结构图保存失败！", e)

    def on_after_test_epoch(self, trainer, task, metrics):
        if not trainer.main_process:
            return
        if trainer.hyper_params is not None:
            self.log_hparams(trainer.hyper_params, metrics)

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.global_train_batch_idx += 1
        if not trainer.main_process:
            return
        self.log(metrics, 'train', 'batch', self.global_train_batch_idx)

    def on_after_train_epoch(self, trainer, task, metrics):
        self.global_train_epoch_idx += 1
        if not trainer.main_process:
            return
        self.log(metrics, 'train', 'epoch', self.global_train_epoch_idx)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.global_val_batch_idx += 1
        if not trainer.main_process:
            return
        self.log(metrics, 'val', 'batch', self.global_val_batch_idx)

    def on_after_val_epoch(self, trainer, task, metrics):
        self.global_val_epoch_idx += 1
        if not trainer.main_process:
            return
        self.log(metrics, 'val', 'epoch', self.global_val_epoch_idx)

    def run_tensorboard(self):
        run_tensorboard(self.log_dir)


def run_tensorboard(log_dir):
    if 'Windows' in platform.system():
        tensorboard = osp.join(sys.prefix, 'Scripts', 'tensorboard')
    else:
        tensorboard = osp.join(sys.prefix, 'bin', 'tensorboard')
    try:
        os.system(f'{tensorboard} --logdir={log_dir}')
    except:  # pylint: disable=W0702
        pass
