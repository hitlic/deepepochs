"""
@author: hitlic
"""
from collections.abc import Iterable
import time
from .loops import log_batch, log_epoch, check_path, StopLoopException, LoopException
from os import path as osp
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


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
    def __init__(self, long_output):
        """
        默认启用的Callback，实现功能：
            指标输出
            学习率调度
        Args:
            long_output: 指标输出为长格式（7位小说）还是短格式（4位小数）
        """
        super().__init__(priority=0)
        self.round_to = 7 if long_output else 4

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
        log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.global_train_batch_idx, self.total_train_batchs, 'TRAIN', self.epoch_width, self.batch_width, self.round_to)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.global_val_batch_idx += 1
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
        log_batch(metrics, self.global_test_epoch_idx, self.total_test_epochs, self.global_test_batch_idx, self.total_test_batchs, 'TEST', self.epoch_width, self.batch_width, self.round_to)


class CheckCallback(Callback):
    def __init__(self, monitor, on_stage='val', mode='min', patience=0, ckpt_dir='./logs'):
        """
        实现功能：Checkpoint和Early Stop。其中，仅保存监控指标最优Checkpoint。
        Args:
            monitor:   监控指标
            on_stage:  监控目标，'train'或'val'
            mode:      监控指标模式，'max'或'min'
            patience:  Early Stop 容忍指标连续变差的次数，0表示不启用Early Stop
            ckpt_dir:  最优模型Checkpoint的保存位置
        """
        self.monitor = monitor
        assert on_stage in ['train', 'val'], 'CheckCallback的`on_stage`参数取值为"train"或"val"'
        self.on_stage = on_stage
        self.mode = mode
        self.patience = patience

        assert mode in ['min', 'max']
        self.best_value = -100000000.0 if  mode == 'max' else 100000000.0

        self.ckpt_dir = ckpt_dir
        self.worse_times = 0
        super().__init__(priority=-1)

    def check(self, metrics, model, opt):
        """
        Reture:
            True:  表示继承执行
            False: 表示达到Early Stop条件
        """
        if self.monitor not in metrics:
            raise LoopException(f'CheckCallback: 要监控的`{self.monitor}`指标不存在！')

        value = metrics[self.monitor]
        if self.mode == 'max':
            if  value > self.best_value:
                self.best_value = value
                save_state(model, opt, self.ckpt_path, best_value=self.best_value)
                self.worse_times = 0
            else:
                self.worse_times += 1
        else:
            if value < self.best_value:
                self.best_value = value
                save_state(model, opt, self.ckpt_path, best_value=self.best_value)
                self.worse_times = 0
            else:
                self.worse_times += 1
        if self.patience > 0 and self.worse_times >= self.patience:
            return False
        return True

    def on_before_fit(self, trainer, epochs):
        if trainer.resume is not False:
            if trainer.resume is True:
                running_id = get_latest_running(self.ckpt_dir)  # 加载最近的checkpoint
            else:
                running_id = str(trainer.resume)                # 加载指定的checkpoint
            try:
                print(f'loading checkpoint of running {running_id} ...')
                path = osp.join(self.ckpt_dir, running_id, 'checkpoint.ckpt')
                self.load_state(trainer, path)
            except FileNotFoundError:
                print('loading failed, checkpoint does not exist!\nstarting training with random parameters!')
            except Exception as e:
                print(f'loading failed! {e}\nstarting training with random parameters!')

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        monitor_metrics = train_metrics if self.on_stage == 'train' else val_metrics
        if self.monitor in monitor_metrics:
            # 创建新的checkpoint路径
            ckpt_dir = osp.join(self.ckpt_dir, trainer.running_id)
            check_path(ckpt_dir)
            self.ckpt_path = osp.join(ckpt_dir, 'checkpoint.ckpt')
            # 检查（保存checkpoint，early stop）
            if not self.check(monitor_metrics, trainer.model, trainer.opt):
                raise StopLoopException(f"Early stopping triggered, by monitoring [{self.on_stage} {self.monitor}]!")
        else:
            raise LoopException(f"CheckCallback: {self.on_stage}阶段的指标中不包含 {self.monitor}!")

    def on_before_test_epochs(self, trainer, tasks):
        try:
            if hasattr(self, 'ckpt_path'):
                print(f'loading best model from running {trainer.running_id} ...')
                self.load_state(trainer, self.ckpt_path)
        except FileNotFoundError as e:
            print('loading best failed,', e)
            print('testing with leatest model.')

    def load_state(self, trainer, ckpt_path):
        other_params = load_state(trainer.model, trainer.opt, ckpt_path)
        for k, v in other_params.items():
            setattr(self, k, v)


def save_state(model, opt, path, **kwargs):
    state = {'model_state': model.state_dict(), 'opt_state': opt.state_dict(), **kwargs}
    torch.save(state, path)


def load_state(model, opt, path):
    state = torch.load(path)
    model.load_state_dict(state['model_state'])
    opt.load_state_dict(state['opt_state'])
    return {k: v for k, v in state.items() if k not in ['model_state', 'opt_state']}


def get_latest_running(from_dir):
    try:
        dir_list = [f for f in os.listdir(from_dir) if osp.isdir(osp.join(from_dir, f))]
        file_list = sorted(dir_list, key=lambda f: osp.getctime(osp.join(from_dir, f)))
        return file_list[-1]
    except Exception:
        return 'no_checkpoint'


class LogCallback(Callback):
    def __init__(self, log_dir='./logs'):
        super().__init__(priority=1)
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
        log_dir = osp.join(self.log_dir, trainer.running_id)
        check_path(log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def on_after_test_epoch(self, trainer, task, metrics):
        if trainer.hyper_params is not None:
            self.log_hparams(trainer.hyper_params, metrics)

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.global_train_batch_idx += 1
        self.log(metrics, 'train', 'batch', self.global_train_batch_idx)

    def on_after_train_epoch(self, trainer, task, metrics):
        self.global_train_epoch_idx += 1
        self.log(metrics, 'train', 'epoch', self.global_train_epoch_idx)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.global_val_batch_idx += 1
        self.log(metrics, 'val', 'batch', self.global_val_batch_idx)

    def on_after_val_epoch(self, trainer, task, metrics):
        self.global_val_epoch_idx += 1
        self.log(metrics, 'val', 'epoch', self.global_val_epoch_idx)

    def run_tensorboard(self):
        os.system(f'tensorboard --logdir={self.log_dir}')
