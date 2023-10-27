"""
@author: hitlic
"""
from typing import List, Dict, Callable
import torch
from torch.optim import Adam
from .loops import *
from collections import defaultdict
from .callbacks import Callback, CallbackPool, DefaultCallback
from torch.utils.data import DataLoader
from datetime import datetime


class EpochTask:
    """一个Epoch的训练、验证或测试任务"""
    def __init__(self, dataloader, metrics=None, do_loss=True, **step_args):
        """
        Args:
            dataloader: pytorch Dataloader
            metrics:    指标函数列表
            do_loss:    验证和测试中是否计算据损失
            args:       其他需要传递给`step`、`train_step`、`val_step`、`test_step`和`evaluate`方法的参数
        """
        self.dataloader = dataloader
        self.batchs = len(dataloader)
        self.metrics = metrics
        self.do_loss = do_loss
        self.trainer = None
        self.stage = None
        self.step_args = step_args

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

        with torch.no_grad():
            self.callbacks.trigger(f'before_{self.stage}_epoch', trainer=self, task=self)
            metrics_values = []
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
                        m_value = step_method(batch_x, batch_y, metrics=self.metrics, do_loss=True, **self.step_args)
                else:
                    m_value = step_method(batch_x, batch_y, metrics=self.metrics, do_loss=self.do_loss, **self.step_args)

                metrics_values.append(m_value)

                # 计算当前batch的指标
                batch_metric_values = flatten_dict(run_patch_dict(m_value), sep='')
                self.callbacks.trigger(f'after_{self.stage}_batch', trainer=self.trainer, metrics=batch_metric_values, batch_idx=batch_idx)
            # 计算当前epoch的指标
            metrics_values = flatten_dict(run_patch_dicts(metrics_values), sep='')
            self.callbacks.trigger(f'after_{self.stage}_epoch', trainer=self.trainer, task=self, metrics=metrics_values)
            return metrics_values


class TrainerBase:
    def __init__(self, model, loss=None, opt=None, epochs=1000, device=None, callbacks=None, metrics=None, resume=False, running_id=None, hyper_params=None):
        """
        Args:
            model:                       Pytorch模型（nn.Module）
            loss:                        损失函数
            opt:                         优化器，或优化器列表；优化器是Pytorch优化器或deepepochs.Optimizer对象
            epochs [int]:                迭代次数
            device [str]:                cpu或cuda
            callbacks [List[Callback]]:  Callback或Callback列表。
            metrics [Callable]:          指标函数列表；通用于训练、验证和测试。
            resume [bool, int, str]:     是否从logs文件平中的Checkpoint加载
                                            - False表示不加载
                                            - True表示从最新的Checkpoint加载
                                            - int、str表示加载相应ID的Checkpoint
            running_id [int, str, None]: 当前训练的运行编号，用于指定日志和checkpoint的文件夹名
            hyper_params [dict, None]:   调参所关注的重要超参数，用于写入日志文件辅助调参
        """
        # 配置损失函数
        if loss is None:
            self.loss = default_loss
        else:
            self.loss = loss

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

        # 迭代次数和device
        self.epochs = epochs
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 模型
        self.model = model.to(self.device)

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

        # 通用于训练、验证和测试的指标
        metrics = [] if metrics is None else metrics
        self.general_metrics = [metrics] if callable(metrics) else list(metrics)

        self.resume = resume  # 该参数会被CheckCallback使用

        if running_id is None:
            self.running_id = str(int(time.time()))  # 以当前时间为running_id
        else:
            self.running_id = str(running_id)
        self.hyper_params = hyper_params

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
        print('=' * 50)
        # print(f'{"DeepEpochs":^50}')
        print(f'training at {datetime.now()}')
        param_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'total parameters {param_size}')
        print('-' * 50)
        assert not (train_dl is None and train_tasks is None), '`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个为None！'
        assert not (train_dl is not None and train_tasks is not None), '`Trainer.fit`方法中，`train_dl`参数和`train_tasks`参数只能有一个不为None！'
 
        # 配置训练与验证指标
        metrics = [] if metrics is None else metrics
        metrics = [metrics] if callable(metrics) else list(metrics)
        metrics = self.general_metrics + [ m for m in metrics if m not in self.general_metrics]  # 使用Trainer.__init__中定义的通用指标
        train_metrics = [] if train_metrics is None else train_metrics
        train_metrics = metrics + [m for m in train_metrics if m not in metrics]
        val_metrics = [] if val_metrics is None else val_metrics
        val_metrics = metrics + [m for m in val_metrics if m not in metrics]

        # 构建训练任务
        train_tasks = [] if train_tasks is None else train_tasks
        train_tasks = [train_tasks] if isinstance(train_tasks, EpochTask) else list(train_tasks)
        if train_dl is not None:
            train_tasks.insert(0, EpochTask(train_dl, metrics=train_metrics))
        for task in train_tasks:
            task.trainer = self

        # 构建验证任务
        val_tasks = [] if val_tasks is None else val_tasks
        val_tasks = [val_tasks] if isinstance(val_tasks, EpochTask) else list(val_tasks)
        if val_dl is not None:
            val_tasks.insert(0, EpochTask(val_dl, metrics=val_metrics, do_loss=do_val_loss))
        for task in val_tasks:
            task.trainer = self

        progress = defaultdict(list)   # 保存各epoch的指标值，作为fit方法返回值

        self.callbacks.trigger('before_fit', trainer=self, epochs=self.epochs)
        try:
            for epoch_idx in range(self.epochs):
                self.callbacks.trigger('before_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, epoch_idx=epoch_idx)
                # training
                train_metric_values = {}
                self.callbacks.trigger('before_train_epochs', trainer=self, tasks=train_tasks, epoch_idx=epoch_idx)
                for train_task in train_tasks:
                    train_task.stage = 'train'
                    train_metric_values.update(train_task())
                self.callbacks.trigger('after_train_epochs', trainer=self, tasks=train_tasks, metrics=train_metric_values, epoch_idx=epoch_idx)
                progress['train'].append(train_metric_values)

                # validation
                val_metric_values = {}
                if val_tasks and (epoch_idx + 1) % val_freq == 0:
                    self.callbacks.trigger('before_val_epochs', trainer=self, tasks=val_tasks, epoch_idx=epoch_idx)
                    for val_task in val_tasks:
                        val_task.stage = 'val'
                        val_metric_values.update(val_task())
                    self.callbacks.trigger('after_val_epochs', trainer=self, tasks=val_tasks, metrics=val_metric_values, epoch_idx=epoch_idx)
                    progress['val'].append(val_metric_values)
                self.callbacks.trigger('after_epoch', trainer=self, train_tasks=train_tasks, val_tasks=val_tasks, train_metrics=train_metric_values, val_metrics=val_metric_values, epoch_idx=epoch_idx)
        except KeyboardInterrupt:
            print('\nStop trainning manually!')
        except StopLoopException as e:
            print('\n', e, sep='')
        except LoopException as e:
            print('\t', e, sep='')

        self.callbacks.trigger('after_fit', trainer=self)
        return {k: concat_dicts(v) for k, v in progress.items()}

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
        print('-'*50)
        metrics = [] if metrics is None else metrics
        metrics = [metrics] if callable(metrics) else list(metrics)
        metrics = self.general_metrics + metrics  # 使用Trainer.__init__中定义的通用指标

        # 构建测试任务
        test_tasks = [] if tasks is None else tasks
        test_tasks = [tasks] if isinstance(tasks, EpochTask) else list(test_tasks)
        if test_dl is not None:
            task = EpochTask(test_dl, metrics=metrics, do_loss=do_loss)
            test_tasks.insert(0, task)
        for task in test_tasks:
            task.trainer = self

        # 运行测试
        try:
            test_metric_values = {}
            self.callbacks.trigger('before_test_epochs', trainer=self, tasks=test_tasks)
            for task in test_tasks:
                task.stage = 'test'
                m_values = task()
                test_metric_values.update(m_values)
            self.callbacks.trigger('after_test_epochs', trainer=self, tasks=test_tasks, metrics=test_metric_values)
            return to_numpy(test_metric_values)
        except LoopException as e:
            print('\n', e, sep='')
        return {}

    def train_step(self, batch_x, batch_y, **step_args) -> Dict[str, PatchBase]:
        """
        TODO: 非常规训练可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据的ValuePatch或者Patch。
        """
        raise NotImplementedError("`Trainer.train_step`方法未实现！")

    def evaluate_step(self,batch_x, batch_y, **step_args) -> Dict[str, PatchBase]:
        """
        TODO: 非常规验证或测试可修改本方法中的代码。也可以定义val_step方法或test_step方法。
        注意：本方法返回一个字典，键为指标名，值为封装了数据的ValuePatch或者Patch。
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
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        self.opt.zero_grad()
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)
        self.callbacks.trigger('before_backward', trainer=self)
        loss.backward()
        self.callbacks.trigger('after_backward', trainer=self)
        self.opt.step()

        results = {'loss': ValuePatch(loss.detach(), len(model_out))}
        for m in step_args.get('metrics', list()):
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results

    def evaluate_step(self,
                      batch_x:[torch.Tensor, List[torch.Tensor]],
                      batch_y:[torch.Tensor, List[torch.Tensor]],
                      **step_args
                      ) -> Dict[str, PatchBase]:
        """
        TODO: 非常规验证或测试可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)
        if step_args.get('do_loss', False):  # 如果存在do_loss参数，且值为True则计算损失
            results = {'loss': ValuePatch(loss.detach(), len(model_out))}
        else:
            results = {}
        for m in step_args.get('metrics', list()):
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results
