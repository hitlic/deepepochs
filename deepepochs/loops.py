"""
@author: hitlic
DeepEpochs is a simple Pytorch deep learning model training tool(see https://github.com/hitlic/deepepochs).
"""
import os
from os import path as osp
import torch
import numpy as np
from typing import Iterable
from copy import deepcopy


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


def batch_size(data):
    if isinstance(data, (list, tuple)):
        return data[0].shape[0]
    elif isinstance(data, torch.Tensor):
        return data.shape[0]
    elif hasattr(data, '__len__'):
        return len(data)
    else:
        return 1


def listify(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, Iterable):
        return list(obj)
    return [obj]


class ddict(dict):
    """
    可以通过“.”访问的字典。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = ddict(v)
                    else:
                        self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = ddict(v)
                else:
                    self[k] = v

    def __getattr__(self, key):
        value = self[key]
        return value

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]

    def __deepcopy__(self, memo=None, _nil=[]):  # pylint: disable=W0102
        dd = dict(self)
        return deepcopy(dd)


class TensorTuple(tuple):
    """
    tuple of tensors
    """
    def __new__(cls, tensors):
        if isinstance(tensors, torch.Tensor):
            tensors=(tensors,)
        return tuple.__new__(cls, tensors)

    @property
    def device(self):
        if len(self) > 0:
            return self[0].device
        else:
            return torch.device(type='cpu')

    def to(self, device, **kwargs):
        return TensorTuple(t.to(device, **kwargs) if isinstance(t, torch.Tensor) else t for t in self)

    def cpu(self):
        return TensorTuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in self)

    def clone(self):
        return TensorTuple(t.clone() if isinstance(t, torch.Tensor) else t for t in self)

    def detach(self):
        return TensorTuple(t.detach() if isinstance(t, torch.Tensor) else t for t in self)

    @property
    def data(self):
        return TensorTuple(t.data if isinstance(t, torch.Tensor) else t for t in self)

    def float(self):
        return TensorTuple(t.float() if isinstance(t, torch.Tensor) else t for t in self)

    def long(self):
        return TensorTuple(t.long() if isinstance(t, torch.Tensor) else t for t in self)

    def int(self):
        return TensorTuple(t.int() if isinstance(t, torch.Tensor) else t for t in self)


def flatten_dict(d, parent_key='', sep='.'):
    """flatten a dict with dict as values"""
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_batch(metrics, epoch_idx, epochs, batch_idx, batchs, stage, epoch_width=0, batch_width=0, round_to=4):
    """
    输出batch指标值
    Args:
        metrics:     指标值字典
        epoch_idx:   当前epoch index
        epochs:      总epochs数量
        batch_idx:   当前batch index
        batchs:      总batch数量
        stage:       train、val或test
        epoch_width: epoch的显示宽度
        batch_width: batch的显示宽度
        round_to:    指标值的小说保留位数
    """
    batch_info = info(metrics, round_to)
    epoch_width = 4 if epoch_width==0 else epoch_width
    batch_width = 4 if batch_width==0 else batch_width
    epoch_idx, epochs = str(epoch_idx).rjust(epoch_width), str(epochs).ljust(epoch_width)
    batch_idx, batchs = str(batch_idx).rjust(batch_width), str(batchs).ljust(batch_width)
    print_out(f'E {epoch_idx}/{epochs}  B {batch_idx}/{batchs}  {stage}> {batch_info}', end='')


def log_epoch(stages_metrics, epoch_idx, epochs, epoch_width=0, round_to=4):
    """输出epoch指标值"""
    epoch_width = 4 if epoch_width==0 else epoch_width
    epoch_idx, epochs = str(epoch_idx).rjust(epoch_width), str(epochs).ljust(epoch_width)

    train_metrics = stages_metrics.get('train')
    val_metrics = stages_metrics.get('val')
    test_metrics = stages_metrics.get('test')
    if train_metrics is not None:
        train_info = info(train_metrics, round_to)
        val_info = ''
        if val_metrics is not None:
            val_info = info(val_metrics, round_to)
            val_info = '  VAL> ' + val_info
        print_out(f'E {epoch_idx}/{epochs}  TRAIN> {train_info}{val_info}')
    elif test_metrics is not None:
        test_info = info(test_metrics, round_to)
        print_out(f'E {epoch_idx}/{epochs}  TEST> {test_info}')
    else:
        raise ValueError("log_epoch 参数错误!")


def info(m_dict, round_to):
    def v_str(v):
        return f'{{:.{round_to}f}}'.format(v).ljust(round_to+3)
    return ' '.join([f'{k:>}: {v_str(v):<}' for k, v in m_dict.items()])


def print_out(content, end='\r\n'):
    print('\x1b[1K\r', end='')
    print(content, flush=True, end=end)


def concat_dicts(dicts, to_np=True):
    if to_np:
        return {k: [to_numpy(d.get(k, 0)) for d in dicts] for k in keyset(dicts)}
    else:
        return {k: [d.get(k, 0) for d in dicts] for k in keyset(dicts)}


def sum_dicts(dicts, to_np=False):
    dicts = concat_dicts(dicts, to_np)
    return ddict({k: sum(v) for k, v in dicts.items()})


def keyset(dicts):
    keys = list(dicts[0].keys())
    keyset = list(set.union(*[set(d.keys()) for d in dicts]))
    for k in keyset:  # 目的是尽量保证key的出现顺序
        if k not in keys:
            keys.append(k)
    return keys


def to_numpy(data):
    """将torch.Tensor或Tensor列表、Tensor字典转为numpy数组"""
    def to(d):
        if isinstance(d, torch.Tensor):
            return d.detach().cpu().numpy()
        else:
            return np.array(d, dtype=float)
    if isinstance(data, (list, tuple)):
        return [to(d) for d in data]
    elif isinstance(data, dict):
        return {k: to(v) for k, v in data.items()}
    else:
        return to(data)


def detach(data):
    """对torch.Tensor或者Tensor列表、Tensor字典进行detach().clone()操作"""
    def to(d):
        if isinstance(d, torch.Tensor):
            return d.detach().clone()
        else:
            return d
    if isinstance(data, (list, tuple)):
        return [to(d) for d in data]
    elif isinstance(data, dict):
        return {k: to(v) for k, v in data.items()}
    else:
        return to(data)


def check_path(path, create=True):
    """检查路径是否存在"""
    if not osp.exists(path):
        if create:
            os.makedirs(path)
        else:
            raise ValueError(f'Path "{path}" does not exists!')
    return path


def default_loss(preds, targets):
    """默认损失函数，直接返回模型预测结果，适用于模型直接返回损失值的情况。"""
    return preds


class StopLoopException(Exception):
    pass


class LoopException(Exception):
    pass


class ModelWrapper:
    """
    用于实现回调：
        on_before_train_forward    on_after_train_forward
        on_before_val_forward      on_after_val_forward
        on_before_test_forward     on_after_test_forward
    """
    def __init__(self, model, trainer):
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.model.load_state_dict(state_dict, strict, assign)


class LossWrapper:
    """
    用于自动完成zero_grad、backward、opt.step等操作
       实现回调： on_before_backward    on_after_backward
    """
    def __init__(self, loss_fn, trainer):
        self.loss_fn = loss_fn
        self.trainer = trainer
        self.stage = None
        self.do_loss = None

    def __call__(self, *args, **kwds):
        if not self.do_loss:
            return None

        if self.stage == 'train':
            self.trainer.opt.zero_grad()
        loss = self.loss_fn(*args, **kwds)
        if self.stage == 'train':
            self.trainer.callbacks.trigger('before_backward', trainer=self, loss=loss)
            loss.backward()
            self.trainer.opt.step()
            self.trainer.callbacks.trigger('after_backward', trainer=self, loss=loss)
        return loss.detach()
