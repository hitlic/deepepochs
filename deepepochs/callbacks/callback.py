"""
@author: hitlic
"""
from collections.abc import Iterable
import time


class CallbackException(Exception):
    pass


class Callback:
    """
    所有Callback的基类。

    方法执行流程：
        on_before_fit
            on_before_epoch
                on_before_train_epochs   # 多个训练任务
                    on_before_train_epoch
                        on_before_train_batch
                            on_before_train_forward
                            on_after_train_forward
                            on_before_train_loss
                                on_before_backward
                                on_after_backward
                            on_after_train_loss
                        on_after_train_batch
                        ...
                    on_after_train_epoch
                ...
                on_after_train_epochs
                on_before_val_epochs     # 多个验证任务
                    on_before_val_epoch
                        on_before_val_batch
                            on_before_val_forward
                            on_after_val_forward
                            on_before_val_loss
                            on_after_val_loss
                        on_after_val_batch
                        ...
                    on_after_val_epoch
                    ...
                on_after_val_epochs
            on_after_epoch
            ...
        on_after_fit
        on_before_test_epochs
            on_before_test_epoch
                on_before_test_batch
                    on_before_test_forward
                    on_after_test_forward
                    on_before_test_loss
                    on_after_test_loss
                on_after_test_batch
                ...
            on_after_test_epoch
            ...
        on_after_test_epochs
    """
    def __init__(self, priority=1):
        """
        Args:
            priority: 任意数值。Callback的优先级，priority值越大before方法越先执行，after方法越后执行。
                      默认取值为时间，即以创建时间为优先级。
        """
        self.priority = priority * time.time()

    def on_before_fit(self, trainer, epochs):
        """
        Args:
            trainer:      Trainer
            epochs: 训练总epochs数
        """

    def on_before_epoch(self, trainer, train_tasks, val_tasks, epoch_idx):
        """
        Args:
            trainer:    Trainer
            train_task: 训练任务
            val_tasks:  验证任务列表
            epoch_idx:  当前训练的epoch index
        """

    def on_before_train_epochs(self, trainer, tasks, epoch_idx):
        """
        Args:
            trainer:   Trainer
            tasks:     训练任务列表
            epoch_idx: 当前训练的epoch index
        """

    def on_before_train_epoch(self, trainer, task):
        """
        Args:
            trainer:      Trainer
            task:         训练任务
            total_batchs: mini-batch总数
        """

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        """
        Args:
            trainer:   Trainer
            batch_x:   当前训练batch模型的输入数据
            batch_y:   当前训练batch的标签
            batch_idx: 当前训练batch index
        """

    def on_before_train_forward(self, trainer):
        """
        Args:
            trainer:   Trainer
        """

    def on_after_train_forward(self, trainer, model_out):
        """
        Args:
            trainer:    Trainer
            model_out:  模型前向预测输出
        """

    def on_before_train_loss(self, trainer, model_out, targets, task):
        """
        Args:
            trainer:    Trainer
            model_out:  模型前向预测输出
            targets:    标签
            task:       当前的EpochTask
        """

    def on_before_backward(self, trainer, loss):
        """
        Args:
            trainer:  Trainer
            loss:     loss
        """

    def on_after_backward(self, trainer, loss):
        """
        Args:
            trainer:  Trainer
            loss:     loss
        """

    def on_after_train_loss(self, trainer, loss, model_out, targets, task):
        """
        Args:
            trainer:    Trainer
            loss:       当前batch的损失
            model_out:  模型前向预测输出
            targets:    标签
            task:       当前的EpochTask
        """

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        """
        Args:
            trainer:   Trainer
            metrics:   当前batch的训练指标值字典
            batch_idx: 当前batch index
        """

    def on_after_train_epoch(self, trainer, task, metrics):
        """
        Args:
            trainer: Trainer
            task: 训练任务
            metrics: 当前epoch的训练指标值字典
        """

    def on_after_train_epochs(self, trainer, tasks, metrics, epoch_idx):
        """
        Args:
            trainer:   Trainer
            tasks:     训练任务列表
            metrics:   训练epoch的验证指标值字典
            epoch_idx: epoch_idx: 当前训练的epoch index
        """

    def on_before_val_epochs(self, trainer, tasks, epoch_idx):
        """
        Args:
            trainer:   Trainer
            tasks: 验证任务列表
            epoch_idx: 当前训练的epoch index
        """

    def on_before_val_epoch(self, trainer, task):
        """
        Args:
            trainer:  Trainer
            total_batchs: mini-batch总数
        """

    def on_before_val_batch(self, trainer, batch_x, batch_y, batch_idx):
        """
        Args:
            trainer:   Trainer
            batch_x:   当前验证batch模型的输入数据
            batch_y:   当前验证batch的标签
            batch_idx: 当前验证batch index
        """

    def on_before_val_forward(self, trainer):
        """
        Args:
            trainer:   Trainer
        """

    def on_after_val_forward(self, trainer, model_out):
        """
        Args:
            trainer:    Trainer
            model_out:  模型前向预测输出
        """

    def on_before_val_loss(self, trainer, model_out, targets, task):
        """
        Args:
            trainer:    Trainer
            model_out:  模型前向预测输出
            targets:    标签
            task:       当前的EpochTask
        """

    def on_after_val_loss(self, trainer, loss, model_out, targets, task):
        """
        Args:
            trainer:    Trainer
            loss:       当前batch的损失
            model_out:  模型前向预测输出
            targets:    标签
            task:       当前的EpochTask
        """

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        """
        Args:
            trainer:   Trainer
            metrics:   当前验证batch的指标值字典
            batch_idx: 当前验证batch index
        """

    def on_after_val_epoch(self, trainer, task, metrics):
        """
        Args:
            trainer: Trainer
            metrics: 验证epoch的验证指标值字典
        """

    def on_after_val_epochs(self, trainer, tasks, metrics, epoch_idx):
        """
        Args:
            trainer:   Trainer
            metrics:   验证epoch的验证指标值字典
            epoch_idx: epoch_idx: 当前训练的epoch index
        """

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        """
        Args:
            trainer:       Trainer
            train_metrics: 当前epoch的训练指标字典
            val_metrics:   当前epoch的验证指标字典
            epoch_idx:     当前训练epoch index
        """

    def on_after_fit(self, trainer):
        """
        Args:
            trainer:  Trainer
        """

    def on_before_test_epochs(self, trainer, tasks):
        """
        Args:
            trainer:   Trainer
            tasks:     测试任务列表
        """

    def on_before_test_epoch(self, trainer, task):
        """
        Args:
            trainer:      Trainer
            total_batchs: mini-batch总数
        """

    def on_before_test_batch(self, trainer, batch_x, batch_y, batch_idx):
        """
        Args:
            trainer:   Trainer
            batch_x:   当前测试batch模型的输入数据
            batch_y:   当前测试batch的标签
            batch_idx: 当前测试batch index
        """

    def on_before_test_forward(self, trainer):
        """
        Args:
            trainer:   Trainer
        """

    def on_after_test_forward(self, trainer, model_out):
        """
        Args:
            trainer:    Trainer
            model_out:  模型前向预测输出
        """

    def on_before_test_loss(self, trainer, model_out, targets, task):
        """
        Args:
            trainer:    Trainer
            model_out:  模型前向预测输出
            targets:    标签
            task:       当前的EpochTask
        """

    def on_after_test_loss(self, trainer, loss, model_out, targets, task):
        """
        Args:
            trainer:    Trainer
            loss:       当前batch的损失
            model_out:  模型前向预测输出
            targets:    标签
            task:       当前的EpochTask
        """

    def on_after_test_batch(self, trainer, metrics, batch_idx):
        """
        Args:
            trainer:   Trainer
            metrics:   当前测试batch的指标值字典
            batch_idx: 当前测试batch index
        """

    def on_after_test_epoch(self, trainer, task, metrics):
        """
        Args:
            trainer: Trainer
            metrics: 测试epoch的指标值字典
        """

    def on_after_test_epochs(self, trainer, tasks, metrics):
        """
        Args:
            trainer:   Trainer
            tasks:     测试任务列表
            metrics:   测试epochs的验证指标值字典
        """


class CallbackPool(list):
    """
    用于管理、执行Callback方法的类
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare(self):
        self.sort(key=lambda cbk: cbk.priority)

    def append(self, callback: Callback):
        assert isinstance(callback, Callback), '`callback`必须是Callback的子类对象！'
        return super().append(callback)

    def extend(self, callbacks: Iterable):
        assert all(isinstance(cbk, Callback) for cbk in callbacks), '`callbacks`中必须都是Callback的子类对象！'
        return super().extend(callbacks)

    def trigger(self, event, *args, **kwargs):
        if 'before' in event:
            cbk_ids = range(len(self))
        else:
            cbk_ids = range(len(self)-1, -1, -1)
        for i in cbk_ids:
            getattr(self[i], f'on_{event}')(*args, **kwargs)
