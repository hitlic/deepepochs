"""
@author: liuchen
"""
import torch
from contextlib import nullcontext
from ..loops import LoopException, flatten_dict, listify
from ..loops import batch_size as guess_batch_size
from ..patches import PatchBase, run_patch_dict, aggregate_patch_dicts


class EpochTask:
    """一个Epoch的训练、验证或测试任务"""
    def __init__(self, dataloader, metrics=None, do_loss=True, batch_size=None, **step_args):
        """
        Args:
            dataloader:  pytorch Dataloader
            metrics:     指标函数列表
            do_loss:     验证和测试中是否计算据损失
            batch_size:  整数或者函数，用于计算batch_size。
                            若为整数，则在损失和指标计算中直接使用该数值；
                            若为函数，则函数的参数为(batch_x, batch_y)，返回一个整数。
            step_args:   其他需要传递给`step`、`train_step`、`val_step`、`test_step`和`evaluate`方法的参数
        """
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

    def __len__(self):
        return self.batchs

    def __getattr__(self, name):
        """如果要找的属性和方法不存在，则到trainer中找"""
        return getattr(self.trainer, name, None)

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
        dataloader = getattr(self, f'_{self.stage}_dataloader')
        if dataloader is None:
            dataloader = self.trainer.distributed.prepare_dataloader(
                self.dataloader, training=self.stage == 'train', split_batches=self.trainer.split_batches)
        self.batchs = len(dataloader)
        self.trainer.distributed.set_epoch(dataloader, getattr(self.trainer, 'epoch_idx', 0))
        if self.stage != 'train':
            self.trainer.distributed.sync_buffers(self.model.model)

        try:
            with torch.no_grad():
                self.callbacks.trigger(f'before_{self.stage}_epoch', trainer=self.trainer, task=self)
                epoch_patch_dicts = []
                for batch_idx, batch_data in enumerate(dataloader):
                    to_device = not (self.stage == 'train' and not self.trainer.auto_traindata_to_device)
                    batch_x, batch_y = self.prepare_data(batch_data, to_device)
                    self.current_batch_size = self.get_batch_size(batch_x, batch_y)
                    if self.stage == 'train':
                        steps = self.trainer.gradient_accumulation_steps
                        new_group = batch_idx % steps == 0
                        if new_group:
                            self.trainer._accumulation_count = 0
                        self.trainer._accumulation_count += self.current_batch_size
                        self.trainer._accumulation_should_step = (batch_idx + 1) % steps == 0 or batch_idx + 1 == self.batchs
                        self.loss.begin_batch(new_group)
                    self.callbacks.trigger(f'before_{self.stage}_batch', trainer=self.trainer, batch_x=batch_x, batch_y=batch_y, batch_idx=batch_idx)

                    # 获取mini-batch的`*step`方法
                    #    1. 最优先使用`EpochTask.step`、`Trainer.step`
                    step_method = getattr(self, 'step', None)
                    #    2. 次优先使用`EpochTask.train_step`、`Epoch.val_step`、`EpochTask.test_step`
                    #    3. 其次使用`Trainer.train_step`、`Trainer.val_step`、`Trainer.test_step`
                    step_method = getattr(self, f'{self.stage}_step') if step_method is None else step_method
                    #    4. 再次使用`EpochTask.evaluate_step`方法
                    #    5. 最次使用`Trainer.evaluate_step`
                    step_method = getattr(self, f'{phase}_step') if step_method is None else step_method

                    # 运行mini-batch的`*step`方法
                    if self.stage == 'train':
                        sync = nullcontext() if self.trainer._accumulation_should_step else self.trainer.no_sync()
                        with torch.enable_grad(), sync:
                            step_out = step_method(batch_x, batch_y, **self.step_args)
                        self.loss.finish_batch()
                    else:
                        step_out = step_method(batch_x, batch_y, **self.step_args)

                    if step_out is not None:
                        if not isinstance(step_out, dict):
                            raise LoopException(f'{step_method} 方法的返回值必须为字典！')
                        if not all(isinstance(v, PatchBase) for k, v in step_out.items()):
                            raise LoopException(f'{step_method} 方法返回字典的value必须为Patch（deepepochs.PatchBase子类对象）！')
                        patch_dict = step_out
                    else:
                        patch_dict = {}

                    self.batch_patch_dict.update(patch_dict)
                    epoch_patch_dicts.append(self.batch_patch_dict)

                    # 计算当前batch的指标
                    batch_metric_values = flatten_dict(run_patch_dict(self.batch_patch_dict), sep='')
                    self.callbacks.trigger(f'after_{self.stage}_batch', trainer=self.trainer, metrics=batch_metric_values, batch_idx=batch_idx)
                    # 清空 self.batch_patch_dict
                    self.batch_patch_dict = {}

                # 计算当前epoch的指标
                epoch_metrics_values = flatten_dict(aggregate_patch_dicts(epoch_patch_dicts, self.trainer.distributed), sep='')
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
        确定batch_size。如果在fit方法中指定则使用指定的batch_size，否则进行猜测
        """
        if isinstance(self.batch_size_param, int):
            return self.batch_size_param
        elif self.batch_size_param is None:
            return guess_batch_size(batch_x)
        else:
            return self.batch_size_param(batch_x, batch_y)
