"""
@author: hitlic
"""
from collections.abc import Iterable
import time
from .loops import log_batch, log_epoch, check_path, StopLoopException, LoopException, save_state, load_state
from os import path as osp


class Callback:
    """
    执行流程：
        on_before_fit
            on_before_epoch
                on_before_train_epochs   # 多个训练任务
                    on_before_train_epoch
                        on_before_train_batch
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

    def on_before_backward(self, trainer):
        """
        Args:
            trainer:  Trainer
        """

    def on_after_backward(self, trainer):
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
    def __init__(self):
        """
        默认启用的Callback，实现功能：
            指标输出
            学习率调度
        """
        super().__init__(priority=1)

    def on_before_fit(self, trainer, epochs):
        self.total_epochs = epochs

    def on_before_epoch(self, trainer, train_tasks, val_tasks, epoch_idx):
        self.epoch_idx = epoch_idx
        self.total_train_batchs = sum(task.batchs for task in train_tasks)  # 所有训练任务总batch数量
        self.total_val_batchs = sum(task.batchs for task in val_tasks)      # 所有验证任务总batch数量
        self.current_train_batch_idx = 0                                    # 当前训练batch
        self.current_val_batch_idx = 0                                      # 当前验证batch

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.current_train_batch_idx += 1
        log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.current_train_batch_idx, self.total_train_batchs, 'TRAIN')

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.current_val_batch_idx += 1
        log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.current_val_batch_idx, self.total_val_batchs, 'VAL')

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        if val_metrics:
            log_epoch({'train': train_metrics, 'val': val_metrics}, epoch_idx+1, self.total_epochs)
        else:
            log_epoch({'train': train_metrics}, epoch_idx+1, self.total_epochs)

        # 根据调度器的配置改变优化器学习率
        if val_metrics:  # 优先使用验证损失
            sched_loss = val_metrics.get('loss')
        else:
            sched_loss = train_metrics.get('loss')
        trainer.opt.step(at='epoch', loss=sched_loss)

    def on_before_test_epochs(self, trainer, tasks):
        self.total_test_epochs = len(tasks)
        self.current_test_epoch_id = 0
        self.total_test_batchs = sum(task.batchs for task in tasks)
        self.current_test_batch_idx = 0

    def on_after_test_epoch(self, trainer, task, metrics):
        self.current_test_epoch_id += 1
        log_epoch({'test': metrics}, self.current_test_epoch_id, self.total_test_epochs)

    def on_after_test_batch(self, trainer, metrics, batch_idx):
        self.current_test_batch_idx += 1
        log_batch(metrics, self.current_test_epoch_id, self.total_test_epochs, self.current_test_batch_idx, self.total_test_batchs, 'TEST')


class CheckCallback(Callback):
    def __init__(self, monitor, on_stage='val', mode='min', patience=None, path='./logs/checkpoint'):
        """
        实现功能：Checkpoint和Early Stopping
        Args:
            monitor:  监控指标
            on_stage: 监控目标，'train'或'val'
            mode:     监控指标模式，'max'或'min'
            patience: Early Stopping 容忍指标连续变差的次数
            path:     最优模型参数保存位置
        """
        self.monitor = monitor
        assert on_stage in ['train', 'val'], 'CheckCallback的`on_stage`参数取值为"train"或"val"'
        self.on_stage = on_stage
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

    def check(self, metrics, model, opt):
        if self.monitor not in metrics:
            raise LoopException(f'CheckCallback: 要监控的{self.monitor}指标不存在！')
        value = metrics[self.monitor]
        if self.mode == 'max':
            if  value > self.best_value:
                self.best_value = value
                save_state(model, opt, self.path, best_value=self.best_value)
                self.worse_times = 0
            else:
                self.worse_times += 1
        else:
            if value < self.best_value:
                self.best_value = value
                save_state(model, opt, self.path, best_value=self.best_value)
                self.worse_times = 0
            else:
                self.worse_times += 1
        if self.patience is not None and self.worse_times >= self.patience:
            return False
        return True

    def on_before_fit(self, trainer, epochs):
        if trainer.resume:
            try:
                self.load_state(trainer)
            except Exception:
                print('Loading failed, starting training with random parameters!')

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        monitor_metrics = train_metrics if self.on_stage == 'train' else val_metrics
        if self.monitor in monitor_metrics:
            if not self.check(monitor_metrics, trainer.model, trainer.opt):
                raise StopLoopException(f"Early stopping triggered, by monitoring [{self.on_stage} {self.monitor}]!")
        else:
            raise LoopException(f"CheckCallback: {self.on_stage}阶段的指标中不包含 {self.monitor}!")

    def on_before_test_epochs(self, trainer, tasks):
        self.load_state(trainer)

    def load_state(self, trainer):
        print('loading best model ...')
        other_params = load_state(trainer.model, trainer.opt, self.path)
        for k, v in other_params.items():
            setattr(self, k, v)
