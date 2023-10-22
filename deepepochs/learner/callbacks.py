"""
@author: hitlic
"""
from collections.abc import Iterable
import time
from ..loops import log_batch, log_epoch, check_path
from os import path as osp
import torch


class StopLoopExcption(Exception):
    pass


class Callback:
    """
    执行流程：
        on_before_fit
            on_before_train_epoch
                on_before_train_batch
                    on_before_backward
                    on_after_backward
                on_after_train_batch
                ...
                on_before_val_epoch
                    on_before_val_batch
                    on_after_val_batch
                    ...
                on_after_val_epoch
            on_after_train_epoch
            ...
        on_after_fit
        on_before_test_epoch
            on_before_test_batch
            on_after_test_batch
            ...
        on_after_test_epoch
    """
    def __init__(self, priority=1):
        """
        Args:
            priority: 任意数值。Callback的优先级，priority值越大before方法越先执行，after方法越后执行。
                      默认取值为时间，即以创建时间为优先级。
        """
        self.priority = priority * time.time()

    def on_before_fit(self, learner, total_epochs, total_train_batchs, total_val_batchs):
        """
        Args:
            learner:            Learner
            total_epochs:       训练总epochs数
            total_train_batchs: 每个训练epoch的batch总数
            total_val_batchs:   验证batch总数
        """

    def on_after_fit(self, learner):
        """
        Args:
            learner:  Learner
        """

    def on_before_train_epoch(self, learner, epoch_idx):
        """
        Args:
            learner:   Learner
            epoch_idx: 当前训练的epoch index
        """

    def on_after_train_epoch(self, learner, train_metrics, val_metrics, epoch_idx):
        """
        Args:
            learner:       Learner
            train_metrics: 当前epoch的训练指标字典
            val_metrics:   当前epoch的验证指标字典
            epoch_idx:     当前训练epoch index
        """

    def on_before_train_batch(self, learner, batch_x, batch_y, batch_idx):
        """
        Args:
            learner:   Learner
            batch_x:   当前训练batch模型的输入数据
            batch_y:   当前训练batch的标签
            batch_idx: 当前训练batch index
        """

    def on_after_train_batch(self, learner, metrics, batch_idx):
        """
        Args:
            learner:   Learner
            metrics:   当前batch的训练指标字典
            batch_idx: 当前batch index
        """

    def on_before_val_epoch(self, learner):
        """
        Args:
            learner:  Learner
        """

    def on_after_val_epoch(self, learner, metrics):
        """
        Args:
            learner: Learner
            metrics: 验证epoch的验证指标字典
        """

    def on_before_val_batch(self, learner, batch_x, batch_y, batch_idx):
        """
        Args:
            learner:   Learner
            batch_x:   当前验证batch模型的输入数据
            batch_y:   当前验证batch的标签
            batch_idx: 当前验证batch index
        """

    def on_after_val_batch(self, learner, metrics, batch_idx):
        """
        Args:
            learner:   Learner
            metrics:   当前验证batch的指标字典
            batch_idx: 当前验证batch index
        """

    def on_before_test_epoch(self, learner, total_batchs):
        """
        Args:
            learner:      Learner
            total_batchs: 测试batch总数
        """

    def on_after_test_epoch(self, learner, metrics):
        """
        Args:
            learner: Learner
            metrics: 测试epoch的指标字典
        """

    def on_before_test_batch(self, learner, batch_x, batch_y, batch_idx):
        """
        Args:
            learner:   Learner
            batch_x:   当前测试batch模型的输入数据
            batch_y:   当前测试batch的标签
            batch_idx: 当前测试batch index
        """

    def on_after_test_batch(self, learner, metrics, batch_idx):
        """
        Args:
            learner:  Learner
            metrics:   当前测试batch的指标字典
            batch_idx: 当前测试batch index
        """

    def on_before_backward(self, learner):
        """
        Args:
            learner:  Learner
        """

    def on_after_backward(self, learner):
        """
        Args:
            learner:  Learner
        """


class CallbackPool(list):
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
    def __init__(self, ):
        """
        实现功能：指标输出
        """
        super().__init__(priority=1)

    def on_before_fit(self, learner, total_epochs, total_train_batchs, total_val_batchs):
        self.total_epochs = total_epochs
        self.total_train_batchs = total_train_batchs
        self.total_val_batchs = total_val_batchs

    def on_before_train_epoch(self, learner, epoch_idx):
        self.epoch_idx = epoch_idx

    def on_after_train_epoch(self, learner, train_metrics, val_metrics, epoch_idx):
        if val_metrics is not None:
            log_epoch({'train': train_metrics, 'val': val_metrics}, epoch_idx+1, self.total_epochs)
        else:
            log_epoch({'train': train_metrics}, epoch_idx+1, self.total_epochs)

    def on_after_train_batch(self, learner, metrics, batch_idx):
        log_batch(metrics, self.epoch_idx+1, self.total_epochs, batch_idx+1, self.total_train_batchs, 'TRAIN')

    def on_after_val_batch(self, learner, metrics, batch_idx):
        log_batch(metrics, self.epoch_idx+1, self.total_epochs, batch_idx+1, self.total_val_batchs, 'VAL')

    def on_before_test_epoch(self, learner, total_batchs):
        self.total_test_batchs = total_batchs

    def on_after_test_epoch(self, learner, metrics):
        log_epoch({'test': metrics}, 1, 1)

    def on_after_test_batch(self, learner, metrics, batch_idx):
        log_batch(metrics, 1, 1, batch_idx+1, self.total_test_batchs, 'TEST')


class CheckCallback(Callback):
    def __init__(self, monitor, mode, patience=None, path='./logs/checkpoint'):
        """
        实现功能：Checkpoint和Early Stopping
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience

        assert mode in ['min', 'max']
        if mode == 'max':
            self.best_value = -100000000.0
        else:
            self.best_value = 100000000.0

        check_path(path)
        self.path = osp.join(path, 'model.ckpt')

        self.worse_times = 0
        super().__init__(priority=-1)

    def check(self, metrics, model):
        value = metrics[self.monitor]
        if self.mode == 'max':
            if  value > self.best_value:
                self.best_value = value
                self.save_model(model)
                self.worse_times = 0
            else:
                self.worse_times += 1
        else:
            if value < self.best_value:
                self.best_value = value
                self.save_model(model)
                self.worse_times = 0
            else:
                self.worse_times += 1
        if self.patience is not None and self.worse_times >= self.patience:
            return False
        return True

    def save_model(self, model):
        torch.save(model.state_dict(), self.path)

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path))

    def on_after_train_epoch(self, learner, train_metrics, val_metrics, epoch_idx):
        if val_metrics is not None:
            if not self.check(val_metrics, learner.model):
                raise StopLoopExcption("Early stopping triggered!")

    def on_before_test_epoch(self, learner, total_batchs):
        print('loading best model ...')
        self.load_best(learner.model)
