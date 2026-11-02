"""
@author: liuchen
"""
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from itertools import islice
from typing import Callable, Dict, List, Literal, Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks import AnalyzeCallback, CallbackPool, DefaultCallback
from ..callbacks.check import (
    CheckCallback,
    finalize_checkpoints,
    load_evaluation_model,
    prepare_fit_state,
    restore_fit_random_state,
    restore_fit_state,
    restore_test_model,
)
from ..distributed import DistributedContext
from ..loops import (
    LoopException,
    StopLoopException,
    _clear_progress_line,
    _color,
    _ConsoleProgress,
    _format_duration,
    _format_learning_rates,
    _terminal_columns,
    compose_data,
    concat_dicts,
    default_loss,
    listify,
    to_numpy,
)
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
                 non_blocking=False,
                 compile_model=False,
                 precision="fp32",
                 split_batches=False,
                 gradient_accumulation_steps=1,
                 ddp_kwargs=None,
                 compute_batch_metrics=True,
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
            non_blocking [bool]:                自动搬运输入和标签时透传至Tensor.to，默认False
            precision [str]:                    fp32（默认）、fp16 或 bf16；FP16 在 CUDA 上使用 GradScaler
            split_batches [bool]:               将 DataLoader 的批量平均分给各进程，默认 False（每进程批量）
            gradient_accumulation_steps [int]:   每次参数更新累积的 DataLoader 批次数，默认 1
            ddp_kwargs [dict]:                   直接传入 PyTorch DistributedDataParallel 的参数
            compute_batch_metrics [bool]:        是否逐批求值损失以外的指标；False仅在任务结束求值，默认True
            compile_model [bool]:               利用PyTorch 2.x对模型compile以提升速度（暂不支持mps、Windows [v2.1]）
        """
        self.show_info = show_info
        self.current_task = None
        self.auto_traindata_to_device = auto_traindata_to_device
        self.non_blocking = non_blocking

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
            self.scaler = torch.cuda.amp.GradScaler(enabled=precision == 'fp16')   # type: ignore
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
            parameters = list(model.parameters())
            # 推理模型可以没有参数，优化器仅在有参数时自动构造。
            self.opt = Optimizers([Optimizer(Adam(parameters, lr=0.001))] if parameters else [])
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
        for ordinal, callback in enumerate(callbacks):
            callback._callback_ordinal = ordinal
        self.callbacks = CallbackPool(callbacks)
        self.callbacks.prepare()

        # 通用于训练、验证和测试的指标
        self.general_metrics = listify(metrics)
        self.compute_batch_metrics = compute_batch_metrics

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
        from .. import __version__

        self.print('=' * 50)
        self.print(_color(f'DeepEpochs {__version__}', 'heading'))
        self.print(datetime.now())
        self.print(f"{'device:':<12} {self.device}")
        if self.distributed.is_distributed:
            self.print(f"{'processes:':<12} {self.world_size} (DDP)")
        self.print(f"{'precision:':<12} {self.precision}")
        if self.gradient_accumulation_steps > 1:
            self.print(f"{'accumulate:':<12} {self.gradient_accumulation_steps}")
        epochs = f'{self.init_epoch + 1}..{self.max_epochs}' if self.init_epoch < self.max_epochs else f'{self.init_epoch} completed (target {self.max_epochs})'  # type: ignore
        self.print(f"{'epochs:':<12} {epochs}")
        self.print(f"{'init lr:':<12} {_format_learning_rates(self.opt, self.default_cbk.round_to)}")
        for index, task in enumerate(self.train_tasks):
            size = getattr(task._train_dataloader, 'batch_size', None)
            if size is not None:
                task_name = f' {task.name or index}' if len(self.train_tasks) > 1 else ''
                self.print(f"{'batch:':<12} {size} per process{task_name}")
        self.print(f"{'running ID:':<12} {self.running_id}")
        param_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"{'parameters:':<12} {param_size:,} (trainable)")
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
            limit_train_batches=None,
            limit_val_batches=None,
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
                                用于损失和可加权指标的统计计数，不改变DataLoader批量；
                                若为整数，每批均使用该固定计数，需与损失均值分母一致；
                                若为函数，则参数为(batch_x, batch_y)，返回当前批次损失均值的分母，例如样本数或有效token数；
                                None按输入推断实际样本数，包含不足整批的尾批。
            train_metrics:  训练指标函数列表；可与metrics参数同时使用
            val_metrics:    验证指标函数列表；可与metrics参数同时使用
            train_tasks:    训练任务（EpochTask对象）列表
            val_tasks:      验证任务（EpochTask对象）列表；当需要在多个验证数据集上进行不同指标的验证时，将数据和指标封装为EpochTask
            epochs:         首次调用设置目标总轮数，再次调用指定额外轮数；None延续既定目标
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
                # 显式续训从已完成轮数计起，不带入中断时尚未执行的计划。
                self.max_epochs = self.init_epoch + epochs

        self.is_fit_run = True

        # 配置训练与验证指标
        metrics = listify(metrics)
        metrics = metrics + [m for m in self.general_metrics if m not in metrics]  # 使用Trainer.__init__中定义的通用指标
        self.train_metrics = [m for m in listify(train_metrics) if m not in metrics] + metrics
        self.val_metrics =  [m for m in listify(val_metrics) if m not in metrics] + metrics

        EpochTask.validate_max_batches(limit_train_batches)
        EpochTask.validate_max_batches(limit_val_batches)
        # 配置训练任务
        train_tasks = list(listify(train_tasks))
        if train_dl is not None:
            train_tasks.insert(0, EpochTask(train_dl, metrics=self.train_metrics, batch_size=batch_size))
        for task in train_tasks:
            task.trainer = self
            task._train_max_batches = task.max_batches if limit_train_batches is None else limit_train_batches
            task._train_dataloader = self.distributed.prepare_dataloader(
                task.dataloader, training=True, split_batches=self.split_batches)

        # 合并仅作用于本次调用的任务列表，调用方可复用原列表。
        val_tasks = list(listify(val_tasks))
        if val_dl is not None:
            val_tasks.insert(0, EpochTask(val_dl, metrics=self.val_metrics, do_loss=do_val_loss, batch_size=batch_size))
        for task in val_tasks:
            task.trainer = self
            task.val_freq = val_freq
            task._val_max_batches = task.max_batches if limit_val_batches is None else limit_val_batches
            task._val_dataloader = self.distributed.prepare_dataloader(
                task.dataloader, training=False, split_batches=self.split_batches)

        self.train_tasks, self.val_tasks = train_tasks, val_tasks

        # 保存各epoch的指标值，作为fit方法返回值
        progress_metrics = defaultdict(list)

        # Fit
        progress_bar = None
        started_at = time.perf_counter()
        try:
            prepare_fit_state(self)
            self.callbacks.trigger('before_fit', trainer=self, epochs=self.max_epochs)
            restore_fit_state(self, train_tasks, val_tasks)
            # 初始化回调可以随机重置模型；开始训练前以主进程状态统一参数和 buffer。
            self.distributed.sync_model(self.model.model)
            restore_fit_random_state(self)
            if self.show_info:
                self._train_info()
            first_epoch = self.init_epoch
            stopped = False
            try:
                epoch_iter = range(self.init_epoch, self.max_epochs)
                if self.log_tqdm and self.main_process and _terminal_columns():
                    progress_bar = _ConsoleProgress(epoch_iter, file=sys.stdout, leave=False,
                                                    bar_format="{percentage:3.0f}%|{bar}| {desc}")
                    self.default_cbk.tqdm_iter = progress_bar
                    epoch_iter = progress_bar
                for epoch_idx in epoch_iter:
                    self.epoch_idx = epoch_idx
                    self.callbacks.trigger('before_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, epoch_idx=epoch_idx)
                    # 训练
                    train_metric_values = {}
                    self.callbacks.trigger('before_train_epochs', trainer=self, tasks=train_tasks, epoch_idx=epoch_idx)
                    for train_task in train_tasks:
                        self.current_task = train_task
                        train_task.stage = 'train'
                        self._merge_task_metrics(train_metric_values, train_task())
                    self.callbacks.trigger('after_train_epochs', trainer=self, tasks=train_tasks, metrics=train_metric_values, epoch_idx=epoch_idx)
                    progress_metrics['train'].append(train_metric_values)

                    # 验证
                    val_metric_values = {}
                    if val_tasks and (epoch_idx + 1) % val_freq == 0:
                        self.callbacks.trigger('before_val_epochs', trainer=self, tasks=val_tasks, epoch_idx=epoch_idx)
                        for val_task in val_tasks:
                            self.current_task = val_task
                            val_task.stage = 'val'
                            self._merge_task_metrics(val_metric_values, val_task())
                        self.callbacks.trigger('after_val_epochs', trainer=self, tasks=val_tasks, metrics=val_metric_values, epoch_idx=epoch_idx)
                        progress_metrics['val'].append(val_metric_values)
                    # 本轮已完成；早停回调抛出停止信号时也应保留正确的下一轮编号。
                    self.init_epoch = epoch_idx + 1
                    try:
                        self.callbacks.trigger('after_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, train_metrics=train_metric_values, val_metrics=val_metric_values, epoch_idx=epoch_idx)  # noqa: E501
                    except BaseException as error:
                        # 完成本轮保存，并保留当前回调的异常或主动停止原因。
                        finalize_checkpoints(self, callback_error=error)
                        raise
                    else:
                        finalize_checkpoints(self)
            except KeyboardInterrupt:
                stopped = True
                self.print(_color('[STOP]', 'warning'), 'Training interrupted by user.')
            except StopLoopException as e:
                stopped = True
                self.print(_color('[STOP]', 'warning'), str(e) or 'Training stopped by callback.')

            elapsed = time.perf_counter() - started_at
            self.callbacks.trigger('after_fit', trainer=self)
            # 日志回调共享writer；训练结束后统一显示一次实际写入目录。
            logger = getattr(self, 'logger', None)
            log_dir = getattr(logger, 'log_dir', None)
            if log_dir is not None:
                self.print(_color('[LOG]', 'heading'), f'TensorBoard: {log_dir}')
            if self.show_info:
                label = _color('[STOPPED]', 'warning') if stopped else _color('[FINISHED]', 'success')
                duration = _color(f'[{_format_duration(elapsed)}]', 'duration')
                self.print(f'{label} epochs: {self.init_epoch - first_epoch}   {duration}')
            return {k: concat_dicts(v) for k, v in progress_metrics.items()}
        finally:
            # 进度条和分析hooks都随本次fit收尾，test使用自己的批次输出。
            if progress_bar is not None:
                progress_bar.close()
            self.default_cbk.tqdm_iter = None
            _clear_progress_line()
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
            batch_x = batch_x.to(self.device, non_blocking=self.non_blocking)
            batch_y = batch_y.to(self.device, non_blocking=self.non_blocking)

        if isinstance(batch_x, (list, tuple)) and len(batch_x)==1:
            batch_x = batch_x[0]

        if isinstance(batch_y, (list, tuple)) and len(batch_y)==1:
            batch_y = batch_y[0]

        return batch_x, batch_y

    def test(self, test_dl: DataLoader=None, metrics:List[Callable]=None, do_loss:bool=True, batch_size:Union[int, Callable]=None, tasks:List[EpochTask]=None, weights='auto', limit_batches=None)-> dict:  # noqa: E501
        """
        Args:
            test_dl:     测试Dataloader
            metrics:     测试指标函数列表
            do_loss:     是否计算测试损失
            batch_size:  整数或者函数，用于计算batch_size。
                            用于损失和可加权指标的统计计数，不改变DataLoader批量；
                                若为整数，每批均使用该固定计数，需与损失均值分母一致；
                            若为函数，则参数为(batch_x, batch_y)，返回当前批次损失均值的分母，例如样本数或有效token数；
                                None按输入推断实际样本数，包含不足整批的尾批。
            tasks:       测试任务（EpochTask对象）列表；当需要在多个测试数据集上进行不同指标的测试时，将数据和指标封装为EpochTask
        """
        assert not (test_dl is None and tasks is None), '`Trainer.test`方法中，`train_dl`参数和`task`参数不能同时为None！'
        # self.print('-'*50)
        # 使用Trainer.__init__中定义的通用指标
        self.test_metrics = [m for m in listify(metrics) if m not in self.general_metrics] + self.general_metrics

        # 合并仅作用于本次调用的任务列表，调用方可复用原列表。
        EpochTask.validate_max_batches(limit_batches)
        test_tasks = list(listify(tasks))
        if test_dl is not None:
            task = EpochTask(test_dl, metrics=self.test_metrics, do_loss=do_loss, batch_size=batch_size)
            test_tasks.insert(0, task)
        for task in test_tasks:
            task.trainer = self
            task._test_max_batches = task.max_batches if limit_batches is None else limit_batches
            task._test_dataloader = self.distributed.prepare_dataloader(
                task.dataloader, training=False, split_batches=self.split_batches)

        # 运行测试
        try:
            load_evaluation_model(self, weights)
            test_metric_values = {}
            self.callbacks.trigger('before_test_epochs', trainer=self, tasks=test_tasks)
            for task in test_tasks:
                self.current_task = task
                task.stage = 'test'
                metric_values = task()
                self._merge_task_metrics(test_metric_values, metric_values)
            self.callbacks.trigger('after_test_epochs', trainer=self, tasks=test_tasks, metrics=test_metric_values)
            return to_numpy(test_metric_values) # type: ignore
        finally:
            restore_test_model(self)

    @staticmethod
    def _merge_task_metrics(metrics, values):
        duplicate = metrics.keys() & values.keys()
        if duplicate:
            raise ValueError(f'任务指标重名：{", ".join(sorted(duplicate))}；请为EpochTask指定不同的name。')
        metrics.update(values)

    def predict(self, dataloader, weights='current', limit_batches=None):
        """逐批产生独立CPU预测；输入不拆标签，多卡各进程返回自己的分片。

        weights可取current、auto、CheckCallback实例或文件。提前退出迭代时
        应关闭生成器，例如使用contextlib.closing，确保临时权重与模式恢复。
        """
        EpochTask.validate_max_batches(limit_batches)
        loader = self.distributed.prepare_dataloader(dataloader, training=False,
                                                     split_batches=self.split_batches)
        modes = [(module, module.training) for module in self.model.modules()]
        stage, task = self.model.stage, self.current_task
        try:
            load_evaluation_model(self, weights)
            self.model.eval()
            self.model.stage = 'predict'
            self.current_task = None
            # 只在入口同步；不同长度或空分片不进行逐批collective。
            self.distributed.sync_model(self.model.model)
            batches = loader if limit_batches is None else islice(loader, limit_batches)
            for batch in batches:
                with torch.no_grad(), self.autocast():
                    inputs = compose_data(batch).to(self.device, non_blocking=self.non_blocking)
                    result = self._prediction_to_cpu(self.predict_step(inputs))
                # grad/autocast上下文不能跨yield影响调用方。
                yield result
        finally:
            try:
                restore_test_model(self)
            finally:
                self.model.stage, self.current_task = stage, task
                for module, training in modes:
                    module.training = training

    def predict_step(self, batch_x):
        """默认整批作为输入；含标签或样本ID的批次可覆盖此方法自行拆分。"""
        if isinstance(batch_x, dict):
            return self.model(**batch_x)
        if isinstance(batch_x, (tuple, list)):
            return self.model(*batch_x)
        return self.model(batch_x)

    @staticmethod
    def _prediction_to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device='cpu', copy=True)
        if isinstance(value, dict):
            return {key: TrainerBase._prediction_to_cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(TrainerBase._prediction_to_cpu(item) for item in value)
        return value

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

    def __enter__(self):
        """在同一个上下文中完成训练与测试，退出时释放资源。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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
        if not self.main_process:
            return
        stream = kwargs.get('file') or sys.stdout
        if self.default_cbk.tqdm_iter is not None:
            with tqdm.external_write_mode(file=stream):
                print(*args, **kwargs)
        else:
            if stream is sys.stdout:
                _clear_progress_line()
            print(*args, **kwargs)

    def train_step(self,
                   batch_x: Union[torch.Tensor, List[torch.Tensor]],
                   batch_y: Union[torch.Tensor, List[torch.Tensor]],
                   **step_args
                   ) -> Dict[str, PatchBase]: # type: ignore
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
        # 训练阶段计算损失、反向传播并请求参数更新；累积窗口中间批次保留梯度。
        self.loss(model_out, batch_y)

    def evaluate_step(self,
                      batch_x: Union[torch.Tensor, List[torch.Tensor]],
                      batch_y: Union[torch.Tensor, List[torch.Tensor]],
                      **step_args
                      ) -> Dict[str, PatchBase]:  # type: ignore
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
        # 评估阶段只计算损失和指标，不反向传播或更新参数。
        self.loss(model_out, batch_y)

    def load_checkpoint(self, ckpt_path):
        """
        从Checkpoint恢复模型、优化器、调度器、GradScaler和已完成轮次。
        """
        from ..callbacks.check import load_state  # pylint: disable=C0415
        load_state(self, ckpt_path)
        self.print(_color('[LOADED]', 'success'), f'checkpoint: {ckpt_path}  next epoch: {self.init_epoch + 1}')
        return self


class Trainer(TrainerBase):
    pass
