"""
@author: liuchen
"""
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Callable, Literal, Union
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator
from ..loops import StopLoopException, LoopException, TensorTuple, default_loss, concat_dicts, to_numpy, listify
from ..optimizer import Optimizer, Optimizers
from ..patches import PatchBase, MeanPatch, TensorPatch, ConfusionPatch
from ..callbacks import CallbackPool, DefaultCallback, CallbackException
from tqdm import tqdm
from .model_wrapper import ModelWrapper
from .loss_wrapper import LossWrapper
from .epoch_task import EpochTask


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
                 ):
        """
        Args:
            model:                              Pytorch模型（nn.Module）
            loss:                               损失函数
            opt:                                优化器，或优化器列表；优化器是Pytorch优化器或deepepochs.Optimizer对象
            epochs [int]:                       迭代次数
            device [str]:                       加速设备，可取值包括
                                                    - cpu、cuda、mps等Pytorch支持的设备
                                                    - Accelerator对象，利用Hugging Face Accelerate实现多机多卡或混合精度训练
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
            compile_model [bool]:               利用PyTorch 2.x对模型compile以提升速度（暂不支持mps、Windows [v2.1]）
        """
        self.show_info = show_info
        self.current_task = None
        self.auto_traindata_to_device = auto_traindata_to_device

        # 检测与配置加速设备
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        # elif torch.backends.mps.is_available() and not compile_model:
        #     self.device = 'mps'
        else:
            self.device = 'cpu'

        # Pytorch支持的设备类型
        device_types = ['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl',
                        'ideep', 'hip', 've', 'fpga', 'ort', 'xla', 'lazy', 'vulkan',
                        'mps', 'meta', 'hpu', 'mtia', 'privateuseone']

        # 使用Accelerate，用于实现分布式或混合精度训练
        if isinstance(self.device, Accelerator):
            self.accelerator = self.device
            self.device = self.accelerator.device
            self.main_process = self.accelerator.is_main_process  # 是否主进程
        else:
            assert str(self.device).split(':', maxsplit=1)[0] in device_types, f'Pytorch不支持的{self.device}设备！\nPytorch支持的设备有：{device_types}'
            self.accelerator = None
            self.main_process = True

        # 配置模型
        if compile_model:
            model = torch.compile(model)
        self.model = ModelWrapper(model, self).to(self.device)

        # 配置损失函数
        if loss is None:
            self.loss = LossWrapper(default_loss, self)
        else:
            self.loss = LossWrapper(loss, self)

        # 配置优化器
        if opt is None:
            self.opt = Optimizer(Adam(model.parameters(), lr=0.001))
        elif isinstance(opt, torch.optim.Optimizer):
            self.opt = Optimizer(opt)
        elif isinstance(opt, (Optimizer, Optimizers)):  # Optimizers是多个Optimizer的列表
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
        callbacks = listify(callbacks)
        self.log_tqdm = log_tqdm
        # log_batch = False if log_tqdm else log_batch
        self.default_cbk = DefaultCallback(log_long, log_batch, log_tqdm)
        callbacks.append(self.default_cbk)  # 自动加入DefaultCallback
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
        self.hyper_params = hyper_params                # 该参数会被LogCallback使用

        self.is_fit_run = False                         # fit方法是否被调用

    def _train_info(self):
        self.print('=' * 50)
        self.print(datetime.now())
        if self.accelerator is None:
            self.print(f"{'device:':<12} {self.device}")
        else:  # Accerate训练下的设备信息
            if self.accelerator.distributed_type == 'NO':  # 单进程Accerate
                self.print(f"{'device:':<12} Accelerate-{self.device}")
            else:   # 分布式训练-类型-进程数量
                self.print(f"{'device:':<12} Accelerate-{self.accelerator.distributed_type}-{self.accelerator.num_processes}")
                self.print(' '*12, "**Note: Training metrics are only calculated in the main process!")
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
        assert not (train_dl is None and train_tasks is None), '`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个为None！'
        assert not (train_dl is not None and train_tasks is not None), '`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个不为None！'

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

        # 配置验证任务
        val_tasks = listify(val_tasks)
        if val_dl is not None:
            val_tasks.insert(0, EpochTask(val_dl, metrics=self.val_metrics, do_loss=do_val_loss, batch_size=batch_size))
        for task in val_tasks:
            task.trainer = self
            task.val_freq = val_freq

        # 保存各epoch的指标值，作为fit方法返回值
        progress_metrics = defaultdict(list)

        # Fit
        self.callbacks.trigger('before_fit', trainer=self, epochs=self.max_epochs)
        try:
            epoch_iter = range(self.init_epoch, self.max_epochs)
            if self.log_tqdm:
                epoch_iter = tqdm(epoch_iter, disable=(not self.main_process), bar_format="{percentage:3.0f}%|{bar}| {desc}")
                self.default_cbk.tqdm_iter = epoch_iter
            for epoch_idx in epoch_iter:
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
                self.callbacks.trigger('after_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, train_metrics=train_metric_values, val_metrics=val_metric_values, epoch_idx=epoch_idx)
                self.init_epoch = epoch_idx + 1
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

    def prepare_data(self, batch_data, to_device):
        """
        划分模型输入和标签，并根据to_device将数据放入GPU或其他加速设备
        Args:
            batch_data: Dataloader的返回的批量数据
            to_device: True表示将数据放入设备，False表示不放入。累积梯度情况下，数据在划分更小的批量后才放进GPU。
        """
        batch_x, batch_y = batch_data[:-1], batch_data[-1:]
        batch_x = [TensorTuple(x) if isinstance(x, (list, tuple)) else x for x in batch_x]
        batch_y = [TensorTuple(y) if isinstance(y, (list, tuple)) else y for y in batch_y]
        batch_x, batch_y = TensorTuple(batch_x), TensorTuple(batch_y)

        if to_device:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        return batch_x, batch_y[0] if len(batch_y)==1 else batch_y

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
        self.print('-'*50)
        # 使用Trainer.__init__中定义的通用指标
        self.test_metrics = [m for m in listify(metrics) if m not in self.general_metrics] + self.general_metrics

        # 配置测试任务
        test_tasks = listify(tasks)
        if test_dl is not None:
            task = EpochTask(test_dl, metrics=self.test_metrics, do_loss=do_loss, batch_size=batch_size)
            test_tasks.insert(0, task)
        for task in test_tasks:
            task.trainer = self

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
        return {}

    def print(self, *args, **kwargs):
        if self.main_process:
            print(*args, **kwargs)

    def train_step(self,
                   batch_x: Union[torch.Tensor, List[torch.Tensor]],
                   batch_y: Union[torch.Tensor, List[torch.Tensor]],
                   **step_args
                   ) -> Dict[str, PatchBase]:
        """
        TODO: 非常规训练可重写本方法
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        # self.model是对Trainer中model参数的封装，
        model_out = self.model(*batch_x)
        # self.loss是对Trainer中loss参数的封装，在训练中会自动调用opt.zero_grad、loss.backward、opt.step等方法
        self.loss(model_out, batch_y)

    def evaluate_step(self,
                      batch_x: Union[torch.Tensor, List[torch.Tensor]],
                      batch_y: Union[torch.Tensor, List[torch.Tensor]],
                      **step_args
                      ) -> Dict[str, PatchBase]:
        """
        TODO: 非常规验证或测试可重写本方法，或定义val_step方法、test_step方法
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
        model_out = self.model(*batch_x)
        # self.loss是对Trainer中loss参数的封装，在训练中会自动调用opt.zero_grad、loss.backward、opt.step等方法
        self.loss(model_out, batch_y)


class Trainer(TrainerBase):
    pass
