"""
@author: liuchen
"""
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from typing import Callable, Dict, List, Literal, Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks import AnalyzeCallback, CallbackException, CallbackPool, DefaultCallback
from ..callbacks.check import CheckCallback, finalize_checkpoints, restore_test_model
from ..distributed import DistributedContext
from ..loops import LoopException, StopLoopException, compose_data, concat_dicts, default_loss, listify, to_numpy
from ..optimizer import Optimizer, Optimizers
from ..patches import ConfusionPatch, MeanPatch, PatchBase, TensorPatch
from .epoch_task import EpochTask
from .loss_wrapper import LossWrapper
from .model_wrapper import ModelWrapper


class TrainerBase:
    def __init__(self, model,
                 loss=None,
                 opt=None,
                 epochs=1000,
                 device=None,
                 callbacks=None,
                 metrics=None,
                 metric_patch: Union[Literal['mean', 'tensor', 'confusion'], PatchBase]='tensor',
                 resume=False,
                 running_id=None,
                 hyper_params=None,
                 log_long=False,
                 log_batch=True,
                 log_tqdm=False,
                 show_info=True,
                 auto_traindata_to_device=True,
                 compile_model=False,
                 precision="fp32",
                 split_batches=False,
                 gradient_accumulation_steps=1,
                 ddp_kwargs=None,
                 ):
        """
        Args:
            model:                              Pytorch模型（nn.Module）
            loss:                               损失函数
            opt:                                优化器，或优化器列表；优化器是Pytorch优化器或deepepochs.Optimizer对象
            epochs [int]:                       迭代次数
            device [str]:                       加速设备，可取值包括
                                                    - cpu、cuda、mps等Pytorch支持的设备
                                                    torchrun 启动时自动使用 Pytorch DDP
            callbacks [List[Callback]]:         Callback或Callback列表。
            metrics [Callable]:                 指标函数列表；通用于训练、验证和测试。
            metric_patch [str, PatchBase]:      封装metrics所用的Patch，取值为字符或PatchBase子类
                                                    若为字符则可选 mean、tensor或confusion。当取值为confusion时，metrics参数中指标必须接受混淆矩阵作为输入
            resume [bool, int, str]:            是否从logs文件平中的Checkpoint加载
                                                    - False表示不加载
                                                    - True表示从最新的Checkpoint加载
                                                    - int、str表示加载相应ID的Checkpoint
            running_id [int, str, None]:        当前训练的运行编号，用于指定日志和checkpoint的文件夹名
            hyper_params [dict, None]:          调参所关注的重要超参数，用于写入日志文件辅助调参
            log_long  [bool]:                   指标输出为长格式（7位小数）还是短格式（4位小数）
            log_batch [bool]:                   训练过程中是否每个batch输出一次指标值
            log_tqdm  [bool]:                   是否使用tqdm显示进度
            show_info [bool]:                   fit时是否显示模型信息
            auto_traindata_to_device:           是否自动将训练数据放入device，在定制step时可根据需要选择
            precision [str]:                    fp32（默认）、fp16 或 bf16；FP16 在 CUDA 上使用 GradScaler
            split_batches [bool]:               将 DataLoader 的批量平均分给各进程，默认 False（每进程批量）
            gradient_accumulation_steps [int]:   每次参数更新累积的 DataLoader 批次数，默认 1
            ddp_kwargs [dict]:                   直接传入 PyTorch DistributedDataParallel 的参数
            compile_model [bool]:               利用PyTorch 2.x对模型compile以提升速度（暂不支持mps、Windows [v2.1]）
        """
        self.show_info = show_info
        self.current_task = None
        self.auto_traindata_to_device = auto_traindata_to_device

        self.distributed = DistributedContext(device)
        self.device = self.distributed.device
        self.main_process = self.distributed.is_main_process
        self.rank = self.distributed.rank
        self.world_size = self.distributed.world_size
        self.split_batches = split_batches
        if precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('precision 必须是 fp32、fp16 或 bf16。')
        if precision == 'fp16' and self.device.type != 'cuda':
            raise ValueError('FP16 训练需要 CUDA；CPU 可使用 fp32 或 bf16。')
        if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
            raise ValueError('gradient_accumulation_steps 必须是正整数。')
        self.precision = precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # PyTorch 2.0/2.1 与较新版本使用不同的 GradScaler 入口。
        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=precision == 'fp16')
        self._accumulation_should_step = True
        self._accumulation_count = 0

        # 单卡和 DDP 共用同一个包装器；验证时由包装器使用原始模型，
        # 避免不同长度的验证分片在 DDP 前向 buffer 同步处等待。
        if compile_model:
            model = torch.compile(model)
        model = self.distributed.wrap_model(model, **(ddp_kwargs or {}))
        self.model = ModelWrapper(model, self)

        # 配置损失函数
        if loss is None:
            self.loss = LossWrapper(default_loss, self)
        else:
            self.loss = LossWrapper(loss, self)
        self.optimize = self.loss.optimize     # 方便在定制的step方法中使用
        self.do_metric = self.loss.do_metric   # 方便在定制的step方法中使用

        # 配置优化器
        if opt is None:
            self.opt = Optimizers([Optimizer(Adam(model.parameters(), lr=0.001))])
        elif isinstance(opt, torch.optim.Optimizer):
            self.opt = Optimizers([Optimizer(opt)])
        elif isinstance(opt, Optimizer):
            self.opt = Optimizers([opt])
        elif isinstance(opt, Optimizers):  # Optimizers是多个Optimizer的列表
            self.opt = opt
        elif isinstance(opt, (list, tuple)):  # 多个优化器的情况
            opt_lst = [Optimizer(o) if isinstance(o, torch.optim.Optimizer) else o for o in opt]
            assert all(isinstance(o, Optimizer) for o in opt_lst), "优化器参数存在错误！"
            self.opt = Optimizers(opt_lst)
        else:
            raise ValueError('`opt`参数取值错误！')

        # 迭代次数
        self.max_epochs = epochs
        # 起始迭代
        self.init_epoch = 0

        # 配置Callbacks
        callbacks = list(listify(callbacks))
        self.log_tqdm = log_tqdm
        # log_batch = False if log_tqdm else log_batch
        self.default_cbk = DefaultCallback(log_long, log_batch, log_tqdm)
        callbacks.append(self.default_cbk)  # 自动加入DefaultCallback
        # 文件编号和恢复匹配沿用用户配置顺序，不依赖回调的时间优先级。
        for ordinal, callback in enumerate(cb for cb in callbacks if isinstance(cb, CheckCallback)):
            callback._checkpoint_ordinal = ordinal
        self.callbacks = CallbackPool(callbacks)
        self.callbacks.prepare()

        # 通用于训练、验证和测试的指标
        self.general_metrics = listify(metrics)

        # 配置指标处理的Patch
        if isinstance(metric_patch, str):
            assert metric_patch in ['mean', 'tensor', 'confusion'], 'metric_patch参数的取值必须为"mean"、"tensor"或"confusion"，或者PatchBase的子类'
            if metric_patch=='mean':
                self.metric_patch = MeanPatch
            elif metric_patch=='tensor':
                self.metric_patch = TensorPatch
            else:
                self.metric_patch = ConfusionPatch
        else:
            assert issubclass(metric_patch, PatchBase), 'metric_patch参数的取值为"mean"、"tensor"或"confusion"，或者PatchBase的子类'
            self.metric_patch = metric_patch

        self.resume = resume                            # 该参数会被CheckCallback使用

        if running_id is None:
            # 以当前timestamp为running_id
            self.running_id = str(int(time.time()*100))
        else:
            self.running_id = str(running_id)
        self.running_id = self.distributed.broadcast_object(self.running_id)
        self.hyper_params = hyper_params                # 该参数会被LogCallback使用

        self.is_fit_run = False                         # fit方法是否被调用

    def _train_info(self):
        self.print('=' * 50)
        self.print(datetime.now())
        self.print(f"{'device:':<12} {self.device}")
        if self.distributed.is_distributed:
            self.print(f"{'processes:':<12} {self.world_size} (DDP)")
        self.print(f"{'running ID:':<12} {self.running_id}")
        param_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"{'parameters:':<12} {param_size}")
        self.print('-' * 50)

    def fit(self,
            train_dl: DataLoader=None,
            val_dl: DataLoader=None,
            metrics: List[Callable]=None,
            val_freq: int=1,
            do_val_loss: bool=True,
            batch_size: Union[int, Callable]=None,
            train_metrics: List[Callable]=None,
            val_metrics: List[Callable]=None,
            train_tasks: List[EpochTask]=None,
            val_tasks: List[EpochTask]= None,
            epochs=None,
            )-> Dict[str, list]:
        """
        训练模型。
            当只有一个验证集时，指定val_dl和相应指标即可；
            当有多个验证集时，先每个数据集和相应指标定义EpochTask，然后传入val_tasks参数。
        Args:
            train_dl:       训练Dataloader
            val_dl:         验证Dataloader
            metrics:        指标函数列表；同时用于训练和验证的。指标函数应当有(预测，标签)两个参数，并返回一个mini-batch的指标均值。
            val_freq:       验证频率
            do_val_loss:    是否计算验证损失
            batch_size:     整数或者函数，用于计算batch_size。
                                若为整数，则在损失和指标计算中直接使用该数值；
                                若为函数，则函数的参数为(batch_x, batch_y)，返回一个整数。
            train_metrics:  训练指标函数列表；可与metrics参数同时使用
            val_metrics:    验证指标函数列表；可与metrics参数同时使用
            train_tasks:    训练任务（EpochTask对象）列表
            val_tasks:      验证任务（EpochTask对象）列表；当需要在多个验证数据集上进行不同指标的验证时，将数据和指标封装为EpochTask
            epochs:         指定当前训练的迭代次数
        """
        # assert not (train_dl is None and train_tasks is None), '`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个为None！'
        # assert not (train_dl is not None and train_tasks is not None), '`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个不为None！'
        if train_dl is None and train_tasks is None:
            raise LoopException('`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个为None！')
        if train_dl is not None and train_tasks is not None:
            raise LoopException('`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个不为None！')

        if epochs is not None:
            if not self.is_fit_run:
                self.max_epochs = epochs            # 首次fit
            else:
                self.max_epochs += epochs           # 再次fit

        if not self.is_fit_run and self.show_info:  # 首次fit输出信息
            self._train_info()

        self.is_fit_run = True

        # 配置训练与验证指标
        metrics = listify(metrics)
        metrics = metrics + [m for m in self.general_metrics if m not in metrics]  # 使用Trainer.__init__中定义的通用指标
        self.train_metrics = [m for m in listify(train_metrics) if m not in metrics] + metrics
        self.val_metrics =  [m for m in listify(val_metrics) if m not in metrics] + metrics

        # 配置训练任务
        train_tasks = listify(train_tasks)
        if train_dl is not None:
            train_tasks.insert(0, EpochTask(train_dl, metrics=self.train_metrics, batch_size=batch_size))
        for task in train_tasks:
            task.trainer = self
            task._train_dataloader = self.distributed.prepare_dataloader(
                task.dataloader, training=True, split_batches=self.split_batches)

        # 合并仅作用于本次调用的任务列表，调用方可复用原列表。
        val_tasks = list(listify(val_tasks))
        if val_dl is not None:
            val_tasks.insert(0, EpochTask(val_dl, metrics=self.val_metrics, do_loss=do_val_loss, batch_size=batch_size))
        for task in val_tasks:
            task.trainer = self
            task.val_freq = val_freq
            task._val_dataloader = self.distributed.prepare_dataloader(
                task.dataloader, training=False, split_batches=self.split_batches)

        # 保存各epoch的指标值，作为fit方法返回值
        progress_metrics = defaultdict(list)

        # Fit
        try:
            self.callbacks.trigger('before_fit', trainer=self, epochs=self.max_epochs)
            # 初始化回调可以随机重置模型；开始训练前以主进程状态统一参数和 buffer。
            self.distributed.sync_model(self.model.model)
            try:
                epoch_iter = range(self.init_epoch, self.max_epochs)
                if self.log_tqdm:
                    epoch_iter = tqdm(epoch_iter, disable=(not self.main_process), bar_format="{percentage:3.0f}%|{bar}| {desc}")
                    self.default_cbk.tqdm_iter = epoch_iter
                for epoch_idx in epoch_iter:
                    self.epoch_idx = epoch_idx
                    self.callbacks.trigger('before_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, epoch_idx=epoch_idx)
                    # 训练
                    train_metric_values = {}
                    self.callbacks.trigger('before_train_epochs', trainer=self, tasks=train_tasks, epoch_idx=epoch_idx)
                    for train_task in train_tasks:
                        self.current_task = train_task
                        train_task.stage = 'train'
                        train_metric_values.update(train_task())
                    self.callbacks.trigger('after_train_epochs', trainer=self, tasks=train_tasks, metrics=train_metric_values, epoch_idx=epoch_idx)
                    progress_metrics['train'].append(train_metric_values)

                    # 验证
                    val_metric_values = {}
                    if val_tasks and (epoch_idx + 1) % val_freq == 0:
                        self.callbacks.trigger('before_val_epochs', trainer=self, tasks=val_tasks, epoch_idx=epoch_idx)
                        for val_task in val_tasks:
                            self.current_task = val_task
                            val_task.stage = 'val'
                            val_metric_values.update(val_task())
                        self.callbacks.trigger('after_val_epochs', trainer=self, tasks=val_tasks, metrics=val_metric_values, epoch_idx=epoch_idx)
                        progress_metrics['val'].append(val_metric_values)
                    # 本轮已完成；早停回调抛出停止信号时也应保留正确的下一轮编号。
                    self.init_epoch = epoch_idx + 1
                    try:
                        self.callbacks.trigger('after_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, train_metrics=train_metric_values, val_metrics=val_metric_values, epoch_idx=epoch_idx)
                    except BaseException as error:
                        # 完成本轮保存，并保留当前回调的异常或主动停止原因。
                        finalize_checkpoints(self, callback_error=error)
                        raise
                    else:
                        finalize_checkpoints(self)
            except KeyboardInterrupt:
                self.print('\nStop trainning manually!')
            except StopLoopException as e:
                self.print('\n', e, sep='')
            except LoopException as e:
                self.print('\t', e, sep='')
            except CallbackException as e:
                self.print('\t', e, sep='')

            self.callbacks.trigger('after_fit', trainer=self)
            return {k: concat_dicts(v) for k, v in progress_metrics.items()}
        finally:
            # 初始化、训练或结束回调失败时，都不能留下捕获后续计算的分析 hooks。
            self._remove_analysis_hooks()

    def prepare_data(self, batch_data, to_device):
        """
        划分模型输入和标签，并根据to_device将数据放入GPU或其他加速设备
        Args:
            batch_data: Dataloader的返回的批量数据
            to_device: True表示将数据放入设备，False表示不放入。累积梯度情况下，数据在划分更小的批量后才放进GPU。
        """
        if isinstance(batch_data, dict):
            batch_x, batch_y = batch_data, [None]
        else:
            if self.loss.loss_fn is default_loss:
                batch_x, batch_y = batch_data[:], [None]
            else:
                batch_x, batch_y = batch_data[:-1], batch_data[-1:]
            batch_x = [compose_data(x) if isinstance(x, (list, tuple)) else x for x in batch_x]
            batch_y = [compose_data(y) if isinstance(y, (list, tuple)) else y for y in batch_y]

        batch_x, batch_y = compose_data(batch_x), compose_data(batch_y)

        if to_device:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        if isinstance(batch_x, (list, tuple)) and len(batch_x)==1:
            batch_x = batch_x[0]

        if isinstance(batch_y, (list, tuple)) and len(batch_y)==1:
            batch_y = batch_y[0]

        return batch_x, batch_y

    def test(self, test_dl: DataLoader=None, metrics:List[Callable]=None, do_loss:bool=True, batch_size:Union[int, Callable]=None, tasks:List[EpochTask]=None)-> dict:
        """
        Args:
            test_dl:     测试Dataloader
            metrics:     测试指标函数列表
            do_loss:     是否计算测试损失
            batch_size:  整数或者函数，用于计算batch_size。
                            若为整数，则在损失和指标计算中直接使用该数值；
                            若为函数，则函数的参数为(batch_x, batch_y)，返回一个整数。
            tasks:       测试任务（EpochTask对象）列表；当需要在多个测试数据集上进行不同指标的测试时，将数据和指标封装为EpochTask
        """
        assert not (test_dl is None and tasks is None), '`Trainer.test`方法中，`train_dl`参数和`task`参数不能同时为None！'
        # self.print('-'*50)
        # 使用Trainer.__init__中定义的通用指标
        self.test_metrics = [m for m in listify(metrics) if m not in self.general_metrics] + self.general_metrics

        # 合并仅作用于本次调用的任务列表，调用方可复用原列表。
        test_tasks = list(listify(tasks))
        if test_dl is not None:
            task = EpochTask(test_dl, metrics=self.test_metrics, do_loss=do_loss, batch_size=batch_size)
            test_tasks.insert(0, task)
        for task in test_tasks:
            task.trainer = self
            task._test_dataloader = self.distributed.prepare_dataloader(
                task.dataloader, training=False, split_batches=self.split_batches)

        # 运行测试
        try:
            test_metric_values = {}
            self.callbacks.trigger('before_test_epochs', trainer=self, tasks=test_tasks)
            for task in test_tasks:
                self.current_task = task
                task.stage = 'test'
                metric_values = task()
                test_metric_values.update(metric_values)
            self.callbacks.trigger('after_test_epochs', trainer=self, tasks=test_tasks, metrics=test_metric_values)
            return to_numpy(test_metric_values)
        except LoopException as e:
            self.print('\n', e, sep='')
        finally:
            restore_test_model(self)
        return {}

    def autocast(self):
        """只包围前向与损失计算，反向传播在此上下文之外执行。"""
        if self.precision == 'fp32':
            return nullcontext()
        dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def no_sync(self):
        """定制梯度累积时包住非末子批的前向和反向；单卡可原样调用。"""
        if self.distributed.is_distributed:
            return self.model.model.no_sync()
        return nullcontext()

    def ddp_data(self):
        """协调 DDP 共享数据准备；所有进程都会执行代码块。

        主进程先完成，其他进程随后执行并复用已有文件或缓存；单进程直接执行。
        """
        return self.distributed.ddp_data()

    def _remove_analysis_hooks(self):
        for callback in self.callbacks:
            if isinstance(callback, AnalyzeCallback):
                callback.remove_hooks()

    def close(self):
        """移除分析 hooks、关闭共享日志writer并释放训练器创建的进程组。"""
        logger = getattr(self, 'logger', None)
        try:
            self._remove_analysis_hooks()
            if logger is not None:
                logger.close()
        finally:
            # fit和test共享日志；结束使用后清空回调引用，重复close不再关闭writer。
            if logger is not None:
                self.logger = None
                for callback in self.callbacks:
                    if getattr(callback, 'logger', None) is logger:
                        callback.logger = None
            self.distributed.close()

    def print(self, *args, **kwargs):
        if self.main_process:
            print(*args, **kwargs)

    def train_step(self,
                   batch_x: Union[torch.Tensor, List[torch.Tensor]],
                   batch_y: Union[torch.Tensor, List[torch.Tensor]],
                   **step_args
                   ) -> Dict[str, PatchBase]:
        """
        非常规训练可重写本方法
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        # self.model是对Trainer中model参数的封装
        if isinstance(batch_x, dict):
            model_out = self.model(**batch_x)
        elif isinstance(batch_x, (list, tuple)):
            model_out = self.model(*batch_x)
        else:
            model_out = self.model(batch_x)
        # self.loss是对Trainer中loss参数的封装，在训练中会自动调用opt.zero_grad、loss.backward、opt.step等方法
        self.loss(model_out, batch_y)

    def evaluate_step(self,
                      batch_x: Union[torch.Tensor, List[torch.Tensor]],
                      batch_y: Union[torch.Tensor, List[torch.Tensor]],
                      **step_args
                      ) -> Dict[str, PatchBase]:
        """
        非常规验证或测试可重写本方法，或定义val_step方法、test_step方法
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        # self.model是对Trainer中model参数的封装
        if isinstance(batch_x, dict):
            model_out = self.model(**batch_x)
        elif isinstance(batch_x, (list, tuple)):
            model_out = self.model(*batch_x)
        else:
            model_out = self.model(batch_x)
        # self.loss是对Trainer中loss参数的封装，在训练中会自动调用opt.zero_grad、loss.backward、opt.step等方法
        self.loss(model_out, batch_y)

    def load_checkpoint(self, ckpt_path):
        """
        从Checkpoint文件中加载模型参数
        """
        from ..callbacks.check import load_state  # pylint: disable=C0415
        print(f'loading checkpoint from {ckpt_path} ...')
        load_state(self, ckpt_path)
        return self


class Trainer(TrainerBase):
    pass
