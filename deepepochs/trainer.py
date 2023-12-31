"""
@author: liuchen
"""
import math
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Callable
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator
from .loops import (StopLoopException, LoopException, TensorTuple,
                    flatten_dict, default_loss, concat_dicts, to_numpy, listify, batch_size, concat, detach_clone)
from .tools import batches
from .optimizer import Optimizer, Optimizers
from .patches import PatchBase, MeanPatch, TensorPatch, run_patch_dict, run_patch_dicts
from .callbacks import CallbackPool, DefaultCallback, CallbackException
from tqdm import tqdm

class EpochTask:
    """一个Epoch的训练、验证或测试任务"""
    def __init__(self, dataloader, metrics=None, do_loss=True, **step_args):
        """
        Args:
            dataloader: pytorch Dataloader
            metrics:    指标函数列表
            do_loss:    验证和测试中是否计算据损失
            step_args:  其他需要传递给`step`、`train_step`、`val_step`、`test_step`和`evaluate`方法的参数
        """
        self.dataloader = dataloader
        self.batchs = len(dataloader)
        self.metrics = listify(metrics)
        self.do_loss = do_loss
        self.trainer = None
        self.stage = None
        self.val_freq = None
        self.step_args = step_args
        self.batch_patch_dict = {}   # 由DefaultCallback中的on_train/val/test_prediction回调注入

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

        # 配置指标，在DefaultCallback中的on_train/val/test_prediction中用于构造Patch
        if self.stage == 'train':
            self.metrics = [m for m in self.metrics if m not in self.train_metrics] + self.train_metrics
        elif self.stage == 'val':
            self.metrics = [m for m in self.metrics if m not in self.val_metrics] + self.val_metrics
        else:
            self.metrics = [m for m in self.metrics if m not in self.test_metrics] + self.test_metrics

        with torch.no_grad():
            self.callbacks.trigger(f'before_{self.stage}_epoch', trainer=self, task=self)
            epoch_patch_dicts = []
            for batch_idx, batch_data in enumerate(self.dataloader):
                batch_x, batch_y = self.prepare_data(batch_data)
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
                    with torch.enable_grad():
                        step_out = step_method(batch_x, batch_y, **self.step_args)
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
            epoch_metrics_values = flatten_dict(run_patch_dicts(epoch_patch_dicts), sep='')
            self.callbacks.trigger(f'after_{self.stage}_epoch', trainer=self.trainer, task=self, metrics=epoch_metrics_values)
            return epoch_metrics_values


class ModelWrapper:
    """
    用于实现回调：
        on_before_train_forward    on_after_train_forward
        on_before_val_forward      on_after_val_forward
        on_before_test_forward     on_after_test_forward
    """
    def __init__(self, model, trainer):
        # self.model = torch.compile(model)
        self.model = model
        self.trainer = trainer
        self.stage = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwds):
        self.trainer.callbacks.trigger(f'before_{self.stage}_forward', trainer=self)
        model_out = self.model(*args, **kwds)
        self.trainer.callbacks.trigger(f'after_{self.stage}_forward', trainer=self, model_out=model_out)
        return model_out

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def parameters(self):
        return self.model.parameters()

    def modules(self):
        return self.model.modules()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class LossWrapper:
    """
    1. 自动完成zero_grad、backward、opt.step等操作
    2. 配合实现梯度累积
    3. 实现回调
            on_before_backward    on_after_backward
            on_before_optimize    on_after_optimize
            on_train_metrics      on_val_metrics       on_test_metrics
    """
    def __init__(self, loss_fn, trainer):
        self.loss_fn = loss_fn
        self.trainer = trainer
        self.stage = None
        self.do_loss = None
        self.task = None

        self.total_loss = 0     # 用于实现累积梯度
        self.model_outs = []    # 用于实现累积梯度
        self.batch_ys = []      # 用于实现累积梯度

    def optimize(self):
        self.trainer.callbacks.trigger('before_optimize', trainer=self)
        self.trainer.opt.step()
        self.trainer.opt.zero_grad()
        self.trainer.callbacks.trigger('after_optimize', trainer=self)

    def __call__(self, model_out, batch_y, grad_accumulate=False):
        """
        Args:
            model_out:         模型预测输出
            batch_y:           标签
            grad_accumulate:   是否累积梯度
        """
        if self.stage == 'train':
            # 计算损失
            loss = self.loss_fn(model_out, batch_y)

            # backward
            self.trainer.callbacks.trigger('before_backward', trainer=self, loss=loss)
            if self.trainer.accelerator is None:
                (loss/self.trainer.grad_accumulate_steps).backward()
            else:       # accelerate的backward
                self.trainer.accelerator.backward(loss/self.trainer.grad_accumulate_steps)
            self.trainer.callbacks.trigger('after_backward', trainer=self, loss=loss)

            # 记录各sub-batch的总损失、模型输出、标签
            _loss = loss.detach().clone()
            self.total_loss += _loss * batch_size(model_out)
            self.model_outs.append(detach_clone(model_out))
            self.batch_ys.append(batch_y)

            # 梯度累积
            if grad_accumulate:
                if self.trainer.accelerator is not None: # DeepEpochs的梯度累积要求仅最后一个sub-batch优化
                    self.optimize()                      # Accelerate的梯度累积要求每个sub-batch都优化
                return _loss
            else:
                self.optimize()
                # 计算平均损失，拼接多次累积度累积中的sub-batch的model_out和batch_y
                loss_4cbk = self.total_loss / sum(batch_size(o) for o in self.model_outs)
                model_out_4cbk = concat(self.model_outs)
                batch_y_4cbk = concat(self.batch_ys)
                self.total_loss = 0
                self.model_outs = []
                self.batch_ys = []
        else:
            # 验证与测试不需要实现分批，如果需要的话可使用较小的batch_size
            model_out_4cbk = model_out
            batch_y_4cbk = batch_y
            if self.do_loss:
                loss_4cbk = self.loss_fn(model_out, batch_y)
            else:
                loss_4cbk = None
        self.trainer.callbacks.trigger(f'{self.stage}_metrics', trainer=self.trainer, loss=loss_4cbk, model_out=model_out_4cbk, batch_y=batch_y_4cbk, task=self.task)
        return loss_4cbk


class TrainerBase:
    def __init__(self, model,
                 loss=None,
                 opt=None,
                 epochs=1000,
                 device=None,
                 callbacks=None,
                 metrics=None,
                 metric_patch:['mean', 'tensor']='tensor',
                 resume=False,
                 running_id=None,
                 hyper_params=None,
                 log_long=False,
                 log_batch=True,
                 log_tqdm=False,
                 show_info=True,
                 compile_model=False,
                 grad_accumulate_steps=1,
                 ):
        """
        Args:
            model:                          Pytorch模型（nn.Module）
            loss:                           损失函数
            opt:                            优化器，或优化器列表；优化器是Pytorch优化器或deepepochs.Optimizer对象
            epochs [int]:                   迭代次数
            device [str]:                   加速设备，可取值包括
                                                - cpu、cuda、mps等Pytorch支持的设备
                                                - Accelerator对象，利用Hugging Face Accelerate实现多机多卡或混合精度训练
            callbacks [List[Callback]]:     Callback或Callback列表。
            metrics [Callable]:             指标函数列表；通用于训练、验证和测试。
            metric_patch [PatchBase]:       封装metrics所用的Patch类型，可选项为 mean 或 tensor
            resume [bool, int, str]:        是否从logs文件平中的Checkpoint加载
                                               - False表示不加载
                                               - True表示从最新的Checkpoint加载
                                               - int、str表示加载相应ID的Checkpoint
            running_id [int, str, None]:    当前训练的运行编号，用于指定日志和checkpoint的文件夹名
            hyper_params [dict, None]:      调参所关注的重要超参数，用于写入日志文件辅助调参
            log_long  [bool]:               指标输出为长格式（7位小数）还是短格式（4位小数）
            log_batch [bool]:               训练过程中是否每个batch输出一次指标值
            log_tqdm  [bool]:               是否使用tqdm显示进度
            compile_model [bool]:           利用PyTorch 2.x对模型compile以提升速度（暂不支持mps、Windows [v2.1]）
            grad_accumulate_steps [int]:    累积梯度更新时的累积次数，大于1表示启用累积梯度更新
        """
        self.show_info = show_info

        # 检测与配置加速设备
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available() and not compile_model:
            self.device = 'mps'
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

        # 梯度累积次数
        assert isinstance(grad_accumulate_steps, int) and grad_accumulate_steps > 0, '梯度累积次数`grad_accumulate_steps`必须为正整数！'
        self.grad_accumulate_steps = grad_accumulate_steps
        if self.accelerator is not None and self.accelerator.gradient_accumulation_steps > 1:
            # 优先使用accelerator中的gradient_accumulation_steps
            self.grad_accumulate_steps = self.accelerator.gradient_accumulation_steps

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
        log_batch = False if log_tqdm else log_batch
        self.default_cbk = DefaultCallback(log_long, log_batch, log_tqdm)
        callbacks.append(self.default_cbk)  # 自动加入DefaultCallback
        self.callbacks = CallbackPool(callbacks)
        self.callbacks.prepare()

        # 通用于训练、验证和测试的指标
        self.general_metrics = listify(metrics)

        # 配置指标处理的Patch（TensorPatch, MeanPatch)
        assert metric_patch in ['mean', 'tensor'], 'metric_patch参数的取值必须为"mean"或"tensor"'
        self.metric_patch = MeanPatch if metric_patch=='mean' else TensorPatch

        self.resume = resume  # 该参数会被CheckCallback使用

        if running_id is None:
            self.running_id = str(int(time.time()))  # 以当前时间为running_id
        else:
            self.running_id = str(running_id)
        self.hyper_params = hyper_params  # 该参数会被LogCallback使用

        self.is_fit_run = False           # fit方法是否被调用

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
            train_metrics:  训练指标函数列表；可与metrics参数同时使用
            val_metrics:    验证指标函数列表；可与metrics参数同时使用
            train_tasks:    训练任务（EpochTask对象）列表
            val_tasks:      验证任务（EpochTask对象）列表；当需要在多个验证数据集上进行不同指标的验证时，将数据和指标封装为EpochTask
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
            train_tasks.insert(0, EpochTask(train_dl, metrics=self.train_metrics))
        for task in train_tasks:
            task.trainer = self

        # 配置验证任务
        val_tasks = listify(val_tasks)
        if val_dl is not None:
            val_tasks.insert(0, EpochTask(val_dl, metrics=self.val_metrics, do_loss=do_val_loss))
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
                    train_task.stage = 'train'
                    train_metric_values.update(train_task())
                self.callbacks.trigger('after_train_epochs', trainer=self, tasks=train_tasks, metrics=train_metric_values, epoch_idx=epoch_idx)
                progress_metrics['train'].append(train_metric_values)

                # 验证
                val_metric_values = {}
                if val_tasks and (epoch_idx + 1) % val_freq == 0:
                    self.callbacks.trigger('before_val_epochs', trainer=self, tasks=val_tasks, epoch_idx=epoch_idx)
                    for val_task in val_tasks:
                        val_task.stage = 'val'
                        val_metric_values.update(val_task())
                    self.callbacks.trigger('after_val_epochs', trainer=self, tasks=val_tasks, metrics=val_metric_values, epoch_idx=epoch_idx)
                    progress_metrics['val'].append(val_metric_values)
                self.callbacks.trigger('after_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, train_metrics=train_metric_values, val_metrics=val_metric_values, epoch_idx=epoch_idx)
                self.init_epoch = epoch_idx + 1
        except KeyboardInterrupt:
            if self.main_process:
                print('\nStop trainning manually!')
        except StopLoopException as e:
            if self.main_process:
                print('\n', e, sep='')
        except LoopException as e:
            if self.main_process:
                print('\t', e, sep='')
        except CallbackException as e:
            if self.main_process:
                print('\t', e, sep='')

        self.callbacks.trigger('after_fit', trainer=self)
        return {k: concat_dicts(v) for k, v in progress_metrics.items()}

    def prepare_data(self, batch_data):
        batch_x, batch_y = TensorTuple(batch_data[:-1]).to(self.device), TensorTuple(batch_data[-1:]).to(self.device)
        return batch_x, batch_y[0] if len(batch_y)==1 else batch_y

    def test(self, test_dl: DataLoader=None, metrics:List[Callable]=None, do_loss:bool=True, tasks:List[EpochTask]=None)-> dict:
        """
        Args:
            test_dl: 测试Dataloader
            metrics: 测试指标函数列表
            do_loss: 是否计算测试损失
            tasks:   测试任务（EpochTask对象）列表；当需要在多个测试数据集上进行不同指标的测试时，将数据和指标封装为EpochTask
        """
        assert not (test_dl is None and tasks is None), '`Trainer.test`方法中，`train_dl`参数和`task`参数不能同时为None！'
        if self.main_process:
            print('-'*50)
        # 使用Trainer.__init__中定义的通用指标
        self.test_metrics = [m for m in listify(metrics) if m not in self.general_metrics] + self.general_metrics

        # 配置测试任务
        test_tasks = listify(tasks)
        if test_dl is not None:
            task = EpochTask(test_dl, metrics=self.test_metrics, do_loss=do_loss)
            test_tasks.insert(0, task)
        for task in test_tasks:
            task.trainer = self

        # 运行测试
        try:
            test_metric_values = {}
            self.callbacks.trigger('before_test_epochs', trainer=self, tasks=test_tasks)
            for task in test_tasks:
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

    def train_step(self, batch_x, batch_y, **step_args):
        """
        TODO: 非常规训练可修改本方法中的代码。
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        raise NotImplementedError("`Trainer.train_step`方法未实现！")

    def evaluate_step(self,batch_x, batch_y, **step_args):
        """
        TODO: 非常规验证或测试可修改本方法中的代码。也可以定义val_step方法或test_step方法。
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        raise NotImplementedError("`Trainer.evaluate_step`方法未实现！")


class Trainer(TrainerBase):
    def train_step(self,
                   batch_x:[torch.Tensor, List[torch.Tensor]],
                   batch_y:[torch.Tensor, List[torch.Tensor]],
                   **step_args
                   ) -> Dict[str, PatchBase]:
        """
        TODO: 非常规训练可修改本方法中的代码。
        Args:
            batch_x:    一个mini-batch的模型输入
            batch_y:    一个mini-batch的标签或targets
            step_args:  当使用EpochTask时，EpochTask的step_args参数
        Returns:
            None 
              或
            dict: 键为指标名，值为封装了数据和指标函数的PatchBase子类对象
        """
        if self.grad_accumulate_steps == 1:
            model_out = self.model(*batch_x)
            # self.loss是对Trainer中loss参数的封装，会自动调用opt.zero_grad、loss.backward、opt.step等方法
            self.loss(model_out, batch_y)
            return

        # 累积梯度训练
        b_size = batch_size(batch_x)
        sub_batch_size = math.ceil(b_size / self.grad_accumulate_steps)
        for sub_batch_idx, (sub_batch_x, sub_batch_y) in enumerate(zip(batches(batch_x, sub_batch_size), batches(batch_y, sub_batch_size))):
            if self.accelerator is None:
                model_out = self.model(*sub_batch_x)
                self.loss(model_out, sub_batch_y, sub_batch_idx + 1 < self.grad_accumulate_steps)
            else:
                with self.accelerator.accumulate(self.model.model):
                    model_out = self.model(*sub_batch_x)
                    self.loss(model_out, sub_batch_y, sub_batch_idx + 1 < self.grad_accumulate_steps)

    def evaluate_step(self,
                      batch_x:[torch.Tensor, List[torch.Tensor]],
                      batch_y:[torch.Tensor, List[torch.Tensor]],
                      **step_args
                      ) -> Dict[str, PatchBase]:
        """
        TODO: 非常规验证或测试可修改本方法中的代码。也可以定义val_step方法或test_step方法。
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
        # self.loss是对Trainer中loss参数的封装，会自动调用opt.zero_grad、loss.backward、opt.step等方法
        self.loss(model_out, batch_y)
