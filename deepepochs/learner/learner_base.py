"""
@author: hitlic
Learner是更完整的Trainer，具有Callback等更复杂的功能。
"""
import torch
from torch.optim import Adam
from typing import Callable, List
from ..loops import *
from collections import defaultdict
from .callbacks import Callback, CallbackPool, StopLoopExcption, DefaultCallback


class LearnerBase:
    def __init__(self, model:torch.nn.Module, loss:Callable=None, opt:torch.optim.Optimizer=None, epochs:int=1000,
                 metrics: List[Callable]=None, device=None, val_freq:int=1,
                 callbacks:List[Callback]=None):
        """
        Args:
            model:      Pytorch模型
            loss:       损失函数
            opt:        优化器
            epochs:     迭代次数
            metrics:    指标
            device:     cpu或cuda
            val_freq:   验证频率
            callbacks:  Callback实体列表
        """
        # 配置损失函数
        if loss is None:
            self.loss = default_loss
        else:
            self.loss = loss
        # 配置优化器
        if opt is None:
            self.opt = Adam(model.parameters(), lr=0.001)
        else:
            self.opt = opt

        self.epochs = epochs        # 迭代次数
        self.metrics = metrics      # 指标
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.val_freq = val_freq    # 验证频率

        # 配置Callbacks
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        callbacks.append(DefaultCallback())  # 自动加入DefaultCallback
        self.callbacks = CallbackPool(callbacks)
        self.callbacks.prepare()

    def fit(self, train_dl, val_dl=None):
        """该方法尽量不要变动"""
        progress = defaultdict(list)   # 保存各epoch的指标值
        self.callbacks.trigger('before_fit', learner=self, total_epochs=self.epochs, total_train_batchs=len(train_dl),
                               total_val_batchs=None if val_dl is None else len(val_dl))
        try:
            for epoch_idx in range(self.epochs):
                self.callbacks.trigger('before_train_epoch', learner=self, epoch_idx=epoch_idx)
                # training
                self.model.train()
                train_metrics = []
                for batch_idx, batch_data in enumerate(train_dl):
                    batch_x, batch_y = self.prepare_data(batch_data)
                    self.callbacks.trigger('before_train_batch', learner=self, batch_x=batch_x, batch_y=batch_y, batch_idx=batch_idx)
                    train_ms = self.train_step(batch_x.to(self.device), None if batch_y is None else batch_y.to(self.device))
                    train_metrics.append(train_ms)
                    with torch.no_grad():
                        # 计算batch指标
                        train_batch_metrics = flatten_dict(run_patch_dict(train_ms), sep='')
                    self.callbacks.trigger('after_train_batch', learner=self, metrics=train_batch_metrics, batch_idx=batch_idx)
                with torch.no_grad():
                    # 计算epoch指标
                    train_metrics = flatten_dict(run_patch_dicts(train_metrics), sep='')
                progress['train'].append(train_metrics)

                # validation
                val_metrics = None
                if val_dl is not None and (epoch_idx + 1) % self.val_freq == 0:
                    self.model.eval()
                    val_metrics = []
                    self.callbacks.trigger('before_val_epoch', learner=self)
                    with torch.no_grad():
                        for batch_idx, batch_data in enumerate(val_dl):
                            batch_x, batch_y = self.prepare_data(batch_data)
                            self.callbacks.trigger('before_val_batch', learner=self, batch_x=batch_x, batch_y=batch_y, batch_idx=batch_idx)
                            val_ms = self.evaluate_step(batch_x.to(self.device), None if batch_y is None else batch_y.to(self.device))
                            val_metrics.append(val_ms)
                            # 计算batch指标
                            val_batch_metrics = flatten_dict(run_patch_dict(val_ms), sep='')
                            self.callbacks.trigger('after_val_batch', learner=self, metrics=val_batch_metrics, batch_idx=batch_idx)
                        # 计算epoch指标
                        val_metrics = flatten_dict(run_patch_dicts(val_metrics), sep='')
                        progress['val'].append(val_metrics)
                    self.callbacks.trigger('after_val_epoch', learner=self, metrics=val_metrics)
                self.callbacks.trigger('after_train_epoch', learner=self, train_metrics=train_metrics, val_metrics=val_metrics, epoch_idx=epoch_idx)
        except KeyboardInterrupt:
            print('\nStop trainning manually!')
        except StopLoopExcption as e:
            print('\n', e, sep='')

        self.callbacks.trigger('after_fit', learner=self)
        return {k: concat_dicts(v) for k, v in progress.items()}

    def prepare_data(self, batch_data):
        batch_x, batch_y = batch_data[:-1], batch_data[-1]
        batch_x = TensorTuple(batch_x)
        if isinstance(batch_y, (list, tuple)):
            batch_y = TensorTuple(batch_y)
        return batch_x, batch_y

    def test(self, test_dl):
        print('-'*30)
        # testing
        self.model.eval()
        test_metrics = []
        batchs = len(test_dl)
        self.callbacks.trigger('before_test_epoch', learner=self, total_batchs=batchs)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_dl):
                batch_x, batch_y = self.prepare_data(batch_data)
                self.callbacks.trigger('before_test_batch', learner=self, batch_x=batch_x, batch_y=batch_y, batch_idx=batch_idx)
                test_m = self.evaluate_step(batch_x.to(self.device), None if batch_y is None else batch_y.to(self.device))
                test_metrics.append(test_m)
                # 计算当前batch的指标
                test_batch_metrics = flatten_dict(run_patch_dict(test_m), sep='')
                self.callbacks.trigger('after_test_batch', learner=self, metrics=test_batch_metrics, batch_idx=batch_idx)
            # 计算当前epoch的指标
            test_metrics = flatten_dict(run_patch_dicts(test_metrics), sep='')
        self.callbacks.trigger('after_test_epoch', learner=self, metrics=test_metrics)
        return to_numpy(test_metrics)

    def train_step(self, batch_x, batch_y):
        """
        TODO: 非常规训练可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据的ValuePatch或者Patch。
        """
        raise NotImplementedError("Trainer.train_step 方法未实现！")

    def evaluate_step(self, batch_x, batch_y):
        """
        TODO: 非常规验证或测试可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据的ValuePatch或者Patch。
        """
        raise NotImplementedError("Trainer.evaluate_step 方法未实现！")
