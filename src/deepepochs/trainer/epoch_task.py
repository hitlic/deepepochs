"""
@author: liuchen
"""
from contextlib import nullcontext
from itertools import islice

import torch

from ..loops import LoopException, flatten_dict, listify
from ..loops import batch_size as guess_batch_size
from ..patches import PatchAccumulator, PatchBase, run_patch_dict


class EpochTask:
    """一个Epoch的训练、验证或测试任务"""
    def __init__(self, dataloader, metrics=None, do_loss=True, batch_size=None, name=None, max_batches=None, **step_args):
        """
        Args:
            dataloader:  pytorch Dataloader
            metrics:     指标函数列表
            do_loss:     验证和测试中是否计算据损失
            batch_size:  整数或者函数，用于计算batch_size。
                            用于损失和可加权指标的统计计数，不改变DataLoader批量；
                            若为整数，每批均使用该固定计数；
                            若为函数，则参数为(batch_x, batch_y)，返回当前批次损失均值的分母；
                            None按输入推断实际样本数。
            name:        任务名称，为对外指标增加name/前缀
            max_batches: 每次执行的本地batch上限；None执行全部
            step_args:   其他需要传递给`step`、`train_step`、`val_step`、`test_step`和`evaluate`方法的参数
        """
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError('name必须是非空字符串或None。')
        self.validate_max_batches(max_batches)
        self.name = name
        self.max_batches = max_batches
        self.dataloader = dataloader
        self.batchs = len(dataloader)
        self.metrics = listify(metrics)
        self.do_loss = do_loss
        self.trainer = None
        self.stage = None
        self.val_freq = None
        self.step_args = step_args
        self.batch_patch_dict = {}   # 由DefaultCallback中的on_train/val/test_metrics回调注入
        if batch_size is not None:
            assert isinstance(batch_size, int) or callable(batch_size), 'batch_size必须是整数或者参数为(batch_x, batch_y)返回整数的函数！'
        self.batch_size_param = batch_size
        self.current_batch_size = None

    @staticmethod
    def validate_max_batches(value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError('batch上限必须是正整数或None。')

    def num_batches(self, stage=None):
        """以当前阶段实际分片和上限计算批数，供尾窗口与日志共用。"""
        stage = self.stage if stage is None else stage
        loader = self.__dict__.get(f'_{stage}_dataloader', self.dataloader)
        limit = self.__dict__.get(f'_{stage}_max_batches', self.max_batches)
        count = len(loader)
        return count if limit is None else min(count, limit)

    def named_metrics(self, metrics):
        return metrics if self.name is None else {f'{self.name}/{key}': value for key, value in metrics.items()}

    def __len__(self):
        return self.batchs

    def __getattr__(self, name):
        """如果要找的属性和方法不存在，则到trainer中找"""
        return getattr(self.trainer, name)

    def __call__(self):
        phase = 'train' if self.stage=='train' else 'evaluate'

        if self.stage == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.model.stage = self.stage
        self.loss.stage = self.stage
        self.loss.do_loss = self.do_loss
        self.loss.task = self

        # 阶段指标只在本次任务执行期间合并，复用同一个 Task 不污染原配置。
        configured_metrics = self.metrics
        stage_metrics = getattr(self, f'{self.stage}_metrics')
        self.metrics = [m for m in configured_metrics if m not in stage_metrics] + stage_metrics
        dataloader = getattr(self, f'_{self.stage}_dataloader', None)
        if dataloader is None:
            dataloader = self.trainer.distributed.prepare_dataloader(
                self.dataloader, training=self.stage == 'train', split_batches=self.trainer.split_batches)
        self.batchs = self.num_batches(self.stage)
        self.trainer.distributed.set_epoch(dataloader, getattr(self.trainer, 'epoch_idx', 0))
        if self.stage != 'train':
            self.trainer.distributed.sync_buffers(self.model.model)

        try:
            with torch.no_grad():
                self.callbacks.trigger(f'before_{self.stage}_epoch', trainer=self.trainer, task=self)
                accumulator = PatchAccumulator(self.trainer.distributed)
                limit = self.__dict__.get(f'_{self.stage}_max_batches', self.max_batches)
                batches = dataloader if limit is None else islice(dataloader, self.batchs)
                for batch_idx, batch_data in enumerate(batches):
                    to_device = not (self.stage == 'train' and not self.trainer.auto_traindata_to_device)
                    batch_x, batch_y = self.prepare_data(batch_data, to_device)
                    self.current_batch_size = self.get_batch_size(batch_x, batch_y)
                    if self.stage == 'train':
                        steps = self.trainer.gradient_accumulation_steps
                        new_group = batch_idx % steps == 0
                        if new_group:
                            self.trainer._accumulation_count = 0
                        self.trainer._accumulation_count += self.current_batch_size   # type: ignore
                        self.trainer._accumulation_should_step = (batch_idx + 1) % steps == 0 or batch_idx + 1 == self.batchs
                        self.loss.begin_batch(new_group)
                    self.callbacks.trigger(f'before_{self.stage}_batch', trainer=self.trainer, batch_x=batch_x, batch_y=batch_y, batch_idx=batch_idx)

                    # 获取mini-batch的`*step`方法
                    #    1. 最优先使用`EpochTask.step`、`Trainer.step`
                    step_method = getattr(self, 'step', None)
                    #    2. 次优先使用`EpochTask.train_step`、`Epoch.val_step`、`EpochTask.test_step`
                    #    3. 其次使用`Trainer.train_step`、`Trainer.val_step`、`Trainer.test_step`
                    step_method = getattr(self, f'{self.stage}_step', None) if step_method is None else step_method
                    #    4. 再次使用`EpochTask.evaluate_step`方法
                    #    5. 最次使用`Trainer.evaluate_step`
                    step_method = getattr(self, f'{phase}_step', None) if step_method is None else step_method

                    # 运行mini-batch的`*step`方法
                    if self.stage == 'train':
                        sync = nullcontext() if self.trainer._accumulation_should_step else self.trainer.no_sync()
                        with torch.enable_grad(), sync:
                            step_out = step_method(batch_x, batch_y, **self.step_args) # type: ignore
                        self.loss.finish_batch()
                    else:
                        step_out = step_method(batch_x, batch_y, **self.step_args) # type: ignore

                    if step_out is not None:
                        if not isinstance(step_out, dict):
                            raise LoopException(f'{step_method} 方法的返回值必须为字典！')
                        if not all(isinstance(v, PatchBase) for k, v in step_out.items()):
                            raise LoopException(f'{step_method} 方法返回字典的value必须为Patch（deepepochs.PatchBase子类对象）！')
                        patch_dict = step_out
                    else:
                        patch_dict = {}

                    self.batch_patch_dict.update(patch_dict)

                    # 完整数据指标可以延后求值；逐批loss仍供日志与学习率查找使用。
                    batch_patches = self.batch_patch_dict
                    if not self.trainer.compute_batch_metrics:
                        batch_patches = {key: patch for key, patch in batch_patches.items() if key == 'loss'}
                    batch_metric_values = self.named_metrics(flatten_dict(run_patch_dict(batch_patches), sep=''))
                    self.callbacks.trigger(f'after_{self.stage}_batch', trainer=self.trainer, metrics=batch_metric_values, batch_idx=batch_idx)
                    accumulator.add(self.batch_patch_dict)
                    # 清空 self.batch_patch_dict
                    self.batch_patch_dict = {}

                # 计算当前epoch的指标
                epoch_metrics_values = self.named_metrics(flatten_dict(accumulator.compute(), sep=''))
                self.callbacks.trigger(f'after_{self.stage}_epoch', trainer=self.trainer, task=self, metrics=epoch_metrics_values)
                return epoch_metrics_values
        except BaseException:
            if self.stage == 'train':
                # 重试从新窗口开始，不能继承中断时尚未提交的梯度与损失。
                self.loss.discard_window()
            raise
        finally:
            self.batch_patch_dict = {}
            self.metrics = configured_metrics

    def get_batch_size(self, batch_x, batch_y):
        """
        确定当前批次用于统计与梯度累积归一化的计数；默认按输入推断样本数。
        """
        if isinstance(self.batch_size_param, int):
            return self.batch_size_param
        elif self.batch_size_param is None:
            return guess_batch_size(batch_x)
        else:
            return self.batch_size_param(batch_x, batch_y)
