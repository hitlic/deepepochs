"""
@author: hitlic
"""
from collections.abc import Iterable
import time
from ..loops import log_batch, log_epoch


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
                            on_before_backward
                            on_after_backward
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

    def on_before_backward(self, trainer, loss):
        """
        Args:
            trainer:  Trainer
        """

    def on_after_backward(self, trainer, loss):
        """
        Args:
            trainer:  Trainer
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


class DefaultCallback(Callback):
    def __init__(self, long_output, log_batch):
        """
        默认启用的Callback，实现功能：
            指标输出
            学习率调度
        Args:
            long_output: 指标输出为长格式（7位小说）还是短格式（4位小数）
            bog_batch:   是否输出batch的指标值
        """
        super().__init__(priority=0)
        self.round_to = 7 if long_output else 4
        self.log_batch = log_batch

    def on_before_fit(self, trainer, epochs):
        self.total_epochs = epochs

    def on_before_epoch(self, trainer, train_tasks, val_tasks, epoch_idx):
        self.epoch_idx = epoch_idx
        self.total_train_batchs = sum(task.batchs for task in train_tasks)  # 所有训练任务总batch数量
        self.total_val_batchs = sum(task.batchs for task in val_tasks)      # 所有验证任务总batch数量
        self.global_train_batch_idx = 0                                     # 当前训练batch
        self.global_val_batch_idx = 0                                       # 当前验证batch

        self.epoch_width = len(str(self.total_epochs))
        self.batch_width = len(str(max(self.total_val_batchs, self.total_train_batchs)))

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.global_train_batch_idx += 1
        if self.log_batch:
            log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.global_train_batch_idx, self.total_train_batchs, 'TRAIN', self.epoch_width, self.batch_width, self.round_to)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.global_val_batch_idx += 1
        if self.log_batch:
            log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.global_val_batch_idx, self.total_val_batchs, 'VAL', self.epoch_width, self.batch_width, self.round_to)

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        if val_metrics:
            log_epoch({'train': train_metrics, 'val': val_metrics}, epoch_idx+1, self.total_epochs, self.epoch_width, self.round_to)
        else:
            log_epoch({'train': train_metrics}, epoch_idx+1, self.total_epochs, self.epoch_width, self.round_to)

        # 根据调度器的配置改变优化器学习率
        if val_metrics:  # 优先使用验证损失
            sched_loss = val_metrics.get('loss')
        else:
            sched_loss = train_metrics.get('loss')
        trainer.opt.step(at='epoch', loss=sched_loss)

    def on_before_test_epochs(self, trainer, tasks):
        self.total_test_epochs = len(tasks)
        self.global_test_epoch_idx = 0
        self.total_test_batchs = sum(task.batchs for task in tasks)
        self.global_test_batch_idx = 0

    def on_after_test_epoch(self, trainer, task, metrics):
        self.global_test_epoch_idx += 1
        log_epoch({'test': metrics}, self.global_test_epoch_idx, self.total_test_epochs, self.epoch_width, self.round_to)

    def on_after_test_batch(self, trainer, metrics, batch_idx):
        self.global_test_batch_idx += 1
        if self.log_batch:
            log_batch(metrics, self.global_test_epoch_idx, self.total_test_epochs, self.global_test_batch_idx, self.total_test_batchs, 'TEST', self.epoch_width, self.batch_width, self.round_to)
