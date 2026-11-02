"""
@author: liuchen
"""
import math
import os
import random as rand
import re
import sys
import time
from copy import deepcopy
from os import path as osp
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm as _tqdm
from tqdm.utils import disp_len as _disp_len


def set_seed(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    rand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def make_runningid(task=None, seed=None, other_info=None, time_precision=1):
    """构造runningid的工具函数"""
    timestamp = str(int(time.time()*time_precision))
    contents = [timestamp]
    if task:
        contents.append(str(task))
    if seed:
        contents.append(str(seed))
    if other_info:
        contents.append(str(other_info))
    return '-'.join(contents)


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


def batch_size(data, batch_dim=0):
    """
    Args:
        data:
        batch_dim: batch所在的维度，默认为batch_first
    """
    if isinstance(data, dict):
        return batch_size(next(iter(data.values())), batch_dim)
    if isinstance(data, (list, tuple)):
        return batch_size(data[0], batch_dim)
    elif isinstance(data, torch.Tensor):
        return 1 if data.numel()==1 else data.shape[batch_dim]
    elif hasattr(data, '__len__'):
        return len(data)
    else:
        return 1  # 数据不是Tensor也没有__len__方法，可能Graph之类的特殊数据


def listify(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (dict, str)):
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


def concat(datas):
    if isinstance(datas[0], (list, tuple)):
        return TensorTuple([torch.concat(ds, dim=0) if ds[0].dim()> 1 else torch.concat(ds) for ds in zip(*datas)])
    elif datas[0] is None:
        return datas
    elif datas[0].numel()==1:
        return torch.stack(datas)
    else:
        return torch.concat(datas, dim=0) if datas[0].dim() > 1 else torch.concat(datas)


def _to(inputs, device, **kwargs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'to'):
        return inputs.to(device, **kwargs)
    return inputs


def _cpu(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'cpu'):
        return inputs.cpu()
    return inputs


def _clone(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'clone'):
        return inputs.clone()
    return inputs


def _detach(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'detach'):
        return inputs.detach()
    return inputs


def _data(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'data'):
        return inputs.data
    return inputs


def _float(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'float'):
        return inputs.float()
    return inputs


def _long(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'long'):
        return inputs.long()
    return inputs


def _int(inputs):
    inputs = compose_data(inputs)
    if isinstance(inputs, torch.Tensor) or hasattr(inputs, 'int'):
        return inputs.int()
    return inputs


class TensorTuple(tuple):
    """
    tuple of tensors
    """
    @property
    def device(self):
        if len(self) > 0:
            return self[0].device
        else:
            return torch.device(type='cpu') # type: ignore

    def to(self, device, **kwargs):
        return TensorTuple(_to(t, device, **kwargs) for t in self)

    def cpu(self):
        return TensorTuple(_cpu(t) for t in self)

    def clone(self):
        return TensorTuple(_clone(t) for t in self)

    def detach(self):
        return TensorTuple(_detach(t) for t in self)

    @property
    def data(self):
        return TensorTuple(_data(t) for t in self)

    def float(self):
        return TensorTuple(_float(t) for t in self)

    def long(self):
        return TensorTuple(_long(t) for t in self)

    def int(self):
        return TensorTuple(_int(t) for t in self)


def tensor_tuple(inputs):
    if isinstance(inputs, (list, tuple)):
        return TensorTuple(inputs)
    else:
        return inputs


class TensorDict(dict):
    """
    dict of tensors
    """
    @property
    def device(self):
        for v in self.values():
            if hasattr(v, 'device'):
                return v.device
        return torch.device(type='cpu') # type: ignore

    def to(self, device, **kwargs):
        return TensorDict({k: _to(v, device, **kwargs) for k, v in self.items()})

    def cpu(self):
        return TensorDict({k: _cpu(v) for k, v in self.items()})

    def clone(self):
        return TensorDict({k: _clone(v) for k, v in self.items()})

    def detach(self):
        return TensorDict({k: _detach(v) for k, v in self.items()})

    @property
    def data(self):
        return TensorDict({k: _data(v) for k, v in self.items()})

    def float(self):
        return TensorDict({k: _float(v) for k, v in self.items()})

    def long(self):
        return TensorDict({k: _long(v) for k, v in self.items()})

    def int(self):
        return TensorDict({k: _int(v) for k, v in self.items()})


def compose_data(data):
    if isinstance(data, (dict)):
        return TensorDict(data)
    elif isinstance(data, (list, tuple)):
        return TensorTuple(data)
    else:
        return data


def clone_value(value):
    """复制一份新的值，包括数值、Tensor或者它们组成的字典、列表或元组"""
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, torch.Tensor):
        return value.detach().clone()
    elif isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    elif isinstance(value, TensorTuple):
        return TensorTuple([clone_value(v) for v in value])
    elif isinstance(value, (list, tuple)):
        return [clone_value(v) for v in value]
    else:
        raise LoopException("Error in value clone!")


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


__PROGRESS_ROWS = 0          # stdout中尚未提交的批次信息占用的物理行数
_ANSI_STYLE = re.compile(r'\x1b\[[0-9;]*m')
_COLOR_CODES = {'train': '36', 'val': '33', 'test': '35',
                'success': '32', 'warning': '33', 'error': '31', 'heading': '1', 'duration': '90'}


def _color(text, role):
    """交互终端只突出标签；NO_COLOR和普通文件使用原始文字。"""
    if not sys.stdout.isatty() or 'NO_COLOR' in os.environ or os.environ.get('TERM') == 'dumb':
        return text
    return f'\x1b[{_COLOR_CODES[role]}m{text}\x1b[0m'


def _format_learning_rates(optimizers, round_to=4):
    """列出各优化器的全部参数组，单一学习率使用简短形式。"""
    rates = [(opt_index, group_index, float(group['lr']))
             for opt_index, optimizer in enumerate(optimizers)
             for group_index, group in enumerate(optimizer.param_groups)]
    if not rates:
        return '-'

    def format_rate(value):
        text = f'{value:.{max(round_to - 1, 0)}e}'
        return text if math.isfinite(value) else _color(text, 'error')

    if len(rates) == 1:
        return format_rate(rates[0][2])
    return ' '.join(f'opt{opt}/group{group}={format_rate(value)}' for opt, group, value in rates)


def _format_duration(seconds):
    if seconds < 60:
        return f'{seconds:.2f}s'
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m{seconds:02d}s' if hours else f'{minutes}m{seconds:02d}s'


def _terminal_columns():
    """只依据当前stdout终端取得列宽；不可用时使用逐行输出。"""
    if not sys.stdout.isatty():
        return 0
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except OSError:
        return 0


def _clear_progress_line():
    """清除当前批次占用的全部物理行，并回到首行开头。"""
    global __PROGRESS_ROWS
    if __PROGRESS_ROWS and sys.stdout.isatty():
        clear = '\r\x1b[2K' + '\x1b[1A\r\x1b[2K' * (__PROGRESS_ROWS - 1)
        print(clear, end='', flush=True)
    __PROGRESS_ROWS = 0


def log_batch(metrics, epoch_idx, epochs, batch_idx, batchs, stage, epoch_width=0, batch_width=0, round_to=4, tqdm_iter=None):
    """刷新当前批次的损失与指标；没有逐批指标时仍显示进度计数。"""
    batch_info = info(metrics, round_to)
    epoch_width = 4 if epoch_width == 0 else epoch_width
    batch_width = 4 if batch_width == 0 else batch_width
    epoch_idx, epochs = str(epoch_idx).rjust(epoch_width), str(epochs).ljust(epoch_width)
    batch_idx, batchs = str(batch_idx).rjust(batch_width), str(batchs).ljust(batch_width)
    stage_label = _color(stage, stage.lower())
    log_info = f'E {epoch_idx}/{epochs}  B {batch_idx}/{batchs}  {stage_label}> {batch_info}'
    print_out(log_info, end='', tqdm_iter=tqdm_iter)


def log_epoch(stages_metrics, epoch_idx, epochs, epoch_width=0, round_to=4, tqdm_iter=None, extra_info=''):
    """将本轮训练/验证或测试指标保留为一行摘要。"""
    if not stages_metrics:
        return
    epoch_width = 4 if epoch_width == 0 else epoch_width
    epoch_idx, epochs = str(epoch_idx).rjust(epoch_width), str(epochs).ljust(epoch_width)

    train_metrics = stages_metrics.get('train')
    val_metrics = stages_metrics.get('val')
    test_metrics = stages_metrics.get('test')
    if train_metrics is not None:
        train_info = info(train_metrics, round_to)
        val_info = ''
        if val_metrics is not None:
            val_info = f'  {_color("VAL", "val")}> ' + info(val_metrics, round_to)
        log_info = f'E {epoch_idx}/{epochs}  {_color("TRAIN", "train")}> {train_info}{val_info}'
    elif test_metrics is not None:
        log_info = f'E {epoch_idx}/{epochs}  {_color("TEST", "test")}>  {info(test_metrics, round_to)}'
    else:
        raise ValueError('log_epoch 参数错误!')
    if extra_info:
        log_info = log_info.rstrip() + '   ' + extra_info
    print_out(log_info, end='\n', tqdm_iter=tqdm_iter)


def info(m_dict, round_to):
    def v_str(value):
        value = value.item() if isinstance(value, torch.Tensor) else value
        if 0 < abs(value) < 10 ** -round_to:
            text = f'{value:.{max(round_to - 1, 0)}e}'
        else:
            text = f'{value:.{round_to}f}'
        padded = text.ljust(round_to + 3)
        return padded if math.isfinite(value) else _color(padded, 'error')
    return ' '.join(f'{key}: {v_str(value)}' for key, value in m_dict.items())


def print_out(content, end='\n', tqdm_iter=None):
    global __PROGRESS_ROWS
    columns = _terminal_columns()
    if not columns:
        # 文件或不可用的终端列宽使用逐行输出，不执行动态覆盖。
        _clear_progress_line()
        print(_ANSI_STYLE.sub('', content), end='\n' if end == '' else end, flush=True)
    elif tqdm_iter is not None:
        if end == '':
            tqdm_iter.set_description_str(content)
        else:
            # 摘要单独写入历史，当前批次描述由下一批重新填充。
            tqdm_iter.set_description_str('', refresh=False)
            tqdm_iter.write(content, end=end, file=tqdm_iter.fp)
    else:
        _clear_progress_line()
        print(('\r' if end == '' else '') + content, end=end, flush=True)
        if end == '':
            # 终端在最右列延迟换行：刚好填满一行时，光标仍在这一行。
            __PROGRESS_ROWS = max(1, (_disp_len(content) - 1) // columns + 1)


class _ConsoleProgress(_tqdm):
    """训练器的进度条与普通批次输出共享同一套物理行清理。"""

    def __str__(self):
        values = self.format_dict.copy()
        values['ncols'] = None
        return self.format_meter(**values)

    @staticmethod
    def status_printer(file):
        def write_status(message):
            if message:
                print_out(message, end='')
            else:
                _clear_progress_line()
        return write_status


def concat_dicts(dicts, to_np=True):
    if to_np:
        return {k: [to_numpy(d.get(k, 0)) for d in dicts] for k in keyset(dicts)}
    else:
        return {k: [d.get(k, 0) for d in dicts] for k in keyset(dicts)}


def sum_dicts(dicts, to_np=False):
    dicts = concat_dicts(dicts, to_np)
    return ddict({k: sum(v) for k, v in dicts.items()})  # type: ignore


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
            d = d.detach().cpu()
            # NumPy 尚无原生 bfloat16；对外数值仍使用 float32 表示。
            if d.dtype == torch.bfloat16:
                d = d.float()
            return d.numpy()
        else:
            return np.array(d, dtype=float)
    if isinstance(data, (list, tuple)):
        return [to(d) for d in data]
    elif isinstance(data, dict):
        return {k: to(v) for k, v in data.items()}
    else:
        return to(data)


def detach_clone(data):
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
    if isinstance(preds, dict) and 'loss' in preds:
        return preds['loss']
    return preds


class StopLoopException(Exception):
    pass


class LoopException(Exception):
    pass
