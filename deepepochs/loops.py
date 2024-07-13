"""
@author: liuchen
"""
import os
from os import path as osp
from copy import deepcopy
from typing import Iterable
import torch
import numpy as np
import random as rand
import time


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


def batch_size(data):
    if isinstance(data, (list, tuple)):
        return batch_size(data[0])
    elif isinstance(data, torch.Tensor):
        return 1 if data.numel()==1 else data.shape[0]
    elif hasattr(data, '__len__'):
        return len(data)
    else:
        return 1


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


class TensorTuple(tuple):
    """
    tuple of tensors
    """
    @property
    def device(self):
        if len(self) > 0:
            return self[0].device
        else:
            return torch.device(type='cpu')

    def to(self, device, **kwargs):
        return TensorTuple(t.to(device, **kwargs) if isinstance(t, torch.Tensor) or hasattr(t, 'to') else t for t in self)

    def cpu(self):
        return TensorTuple(t.cpu() if isinstance(t, torch.Tensor) or hasattr(t, 'cpu') else t for t in self)

    def clone(self):
        return TensorTuple(t.clone() if isinstance(t, torch.Tensor) or hasattr(t, 'clone') else t for t in self)

    def detach(self):
        return TensorTuple(t.detach() if isinstance(t, torch.Tensor) or hasattr(t, 'detach') else t for t in self)

    @property
    def data(self):
        return TensorTuple(t.data if isinstance(t, torch.Tensor) or  hasattr(t, 'data') else t for t in self)

    def float(self):
        return TensorTuple(t.float() if isinstance(t, torch.Tensor) or hasattr(t, 'float') else t for t in self)

    def long(self):
        return TensorTuple(t.long() if isinstance(t, torch.Tensor) or hasattr(t, 'long') else t for t in self)

    def int(self):
        return TensorTuple(t.int() if isinstance(t, torch.Tensor) or hasattr(t, 'int') else t for t in self)


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


__MAX_LOG_INFO_LEN = 0          # 输出的最长字符串
def __update_max_len(log_info): # 更新最长串
    global __MAX_LOG_INFO_LEN
    info_len = len(log_info)
    __MAX_LOG_INFO_LEN = info_len if info_len > __MAX_LOG_INFO_LEN else __MAX_LOG_INFO_LEN


def log_batch(metrics, epoch_idx, epochs, batch_idx, batchs, stage, epoch_width=0, batch_width=0, round_to=4, tqdm_iter=None):
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
    if not metrics:  # metrics为空则不输出
        return
    batch_info = info(metrics, round_to)
    epoch_width = 4 if epoch_width==0 else epoch_width
    batch_width = 4 if batch_width==0 else batch_width
    epoch_idx, epochs = str(epoch_idx).rjust(epoch_width), str(epochs).ljust(epoch_width)
    batch_idx, batchs = str(batch_idx).rjust(batch_width), str(batchs).ljust(batch_width)

    log_info = f'E {epoch_idx}/{epochs}  B {batch_idx}/{batchs}  {stage}> {batch_info}'
    __update_max_len(log_info)
    print_out(log_info, end='', tqdm_iter=tqdm_iter)


def log_epoch(stages_metrics, epoch_idx, epochs, epoch_width=0, round_to=4, tqdm_iter=None):
    """输出epoch指标值
    tqdm_iter: tqdm迭代对象
    """
    if not stages_metrics:  # stages_metrics为空则不输出
        return
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

        log_info = f'E {epoch_idx}/{epochs}  TRAIN> {train_info}{val_info}'
        print_out(log_info, end='\n', tqdm_iter=tqdm_iter)  # 清除光标至行末字符
    elif test_metrics is not None:
        test_info = info(test_metrics, round_to)
        log_info = f'E {epoch_idx}/{epochs}  TEST> {test_info}'
        __update_max_len(log_info)
        print_out(log_info, end='\n', tqdm_iter=tqdm_iter)  # 清除光标至行末字符
    else:
        raise ValueError("log_epoch 参数错误!")


def info(m_dict, round_to):
    def v_str(v):
        v = v.item() if isinstance(v, torch.Tensor) else v
        return f'{{:.{round_to}f}}'.format(v).ljust(round_to+3)
    return ' '.join([f'{k:>}: {v_str(v):<}' for k, v in m_dict.items()])


def print_out(content, end='\n', tqdm_iter=None):
    content += ' ' * (__MAX_LOG_INFO_LEN - len(content))
    if tqdm_iter is None:
        print(end='\r')  # \x1b[1K 清除行首至光标位置字符
        print(content, flush=True, sep='', end=end)
    else:
        tqdm_iter.set_description_str(content)


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
    return preds


class StopLoopException(Exception):
    pass


class LoopException(Exception):
    pass
