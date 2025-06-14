"""
@author: liuchen
一些有用的工具
"""
import math
import itertools
from queue import PriorityQueue
from collections import deque
from itertools import chain
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from .loops import TensorTuple, ddict
import yaml
from functools import reduce
import os
from torch.optim.lr_scheduler import _LRScheduler


class SeriesPlots:
    def __init__(self, plots, colums=1, figsize=None):
        """
        画动态时间序列数据的工具。
        Args:
            plots: 用于指定包含多少个子图，各子图中有几条序列，以及各序列的名字。
                    例如 plots = ['s_name1', ['s_name21', 's_name22']]，表示包含两个子图。
                    第一个子图中有一条名为s_name1的序列，第二个子图中有两条名分别为s_name21和s_name22的序列。
            colums: 子图列数
            fig_size: 图像大小
        Example:
            import random
            import time
            sp = SeriesPlots([ 'x', ['xrand', 'xrand_move']], 2)
            mv = Moveing(10)
            for i in range(1000):
                sp.add(i, [i+random.random()*0.5*i, mv(i+random.random()*0.5*i)])
                time.sleep(0.1)
        """
        self.plot_names = plots
        self.fig_size = figsize
        self.colums = min(len(plots), colums)
        self.rows = 1
        if len(plots) > 1:
            self.rows = math.ceil(len(plots)/colums)

        self.x = []
        self.ys = []
        self.graphs = []

        self.is_start = False
        self.max_ys = [-1000000000] * len(plots)
        self.min_ys = [1000000000] * len(plots)

    def ioff(self):
        """关闭matplotlib交互模式"""
        plt.ioff()

    def start(self):
        plt.ion()
        self.fig, axs = plt.subplots(self.rows, self.colums, figsize=self.fig_size)
        if len(self.plot_names) == 1:
            axs = [axs]
        if self.rows > 1 and self.colums > 1:
            axs = chain(*axs)
        for s_name, ax in zip(self.plot_names, axs):
            if isinstance(s_name, (list, tuple)):
                gs = []
                yy = []
                for n in s_name:
                    ln, = ax.plot([], [], label=n)
                    gs.append((ax, ln))
                    yy.append([])
                self.graphs.append(gs)
                self.ys.append(yy)
            else:
                ln, = ax.plot([], [], label=s_name)
                self.graphs.append((ax, ln))
                self.ys.append([])
            ax.legend()

    def check_shape(self, lstx, lsty):
        spx = [len(item) if isinstance(item, (list, tuple)) else 0 for item in lstx]
        spy = [len(item) if isinstance(item, (list, tuple)) else 0 for item in lsty]
        return spx == spy

    def add(self, *values):
        """
        values的形状必须和 __init__ 中的plots参数形状匹配。
        """
        if not self.is_start:
            self.start()
            self.is_start = True
        assert self.check_shape(values, self.plot_names), f'数据形状不匹配！应当形如：{str(self.plot_names)[1:-1]}'
        self.x.append(len(self.x) + 1)

        ax = None
        for i, (ys, vs, axs_lns) in enumerate(zip(self.ys, values, self.graphs)):
            if isinstance(vs, (list, tuple)):
                self.max_ys[i] = max(self.max_ys[i], max(vs))  # pylint: disable=W3301
                self.min_ys[i] = min(self.min_ys[i], min(vs))  # pylint: disable=W3301
                for y, v, (ax, ln) in zip(ys, vs, axs_lns):
                    y.append(v)
                    ln.set_xdata(self.x)
                    ln.set_ydata(y)
            else:
                self.max_ys[i] = max(self.max_ys[i], vs)
                self.min_ys[i] = min(self.min_ys[i], vs)
                ax, ln = axs_lns
                ys.append(vs)
                ln.set_xdata(self.x)
                ln.set_ydata(ys)
            ax.set_xlim(0, len(self.x) + 1)
            ax.set_ylim(min(0, self.min_ys[i]) - 0.05 * math.fabs(self.min_ys[i]) - 0.1,
                        self.max_ys[i]+0.05*math.fabs(self.max_ys[i]))
            # plt.pause(0.05)
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.05)


def Moveing(length=10):
    """
    返回一个能够计算滑动平均的函数。
    length: 滑动平均长度
    """
    values = deque(maxlen=length)

    def moveing_average(v):
        values.append(v)
        return sum(values)/len(values)
    return moveing_average


class TopKQueue(PriorityQueue):
    """
    能够保存最大k个值的优先队列
    """
    def __init__(self, k: int = 0):
        super().__init__(maxsize=k)

    def put(self, e):
        if self.full():
            if e[0] > self.queue[0][0]:
                self.get()
            else:
                return
        super().put(e)

    def items(self):
        return sorted(self.queue, key=lambda e: e[0], reverse=True)


def batches(inputs, batch_size):
    """
    把inputs按batch_size进行划分.
    Args:
        inputs: Tenor或Tensor列表。注意，不能是列表的列表！
    """
    is_list_input = isinstance(inputs, (list, tuple))  # inputs是否是多个输入组成的列表或元素
    start_idx = 0
    is_over = False
    while True:
        if is_list_input:
            assert all(not isinstance(data, (list, tuple)) for data in inputs), '`inputs`的元素不能是列表或元组！'
            batch = TensorTuple([data[start_idx: start_idx + batch_size] for data in inputs])
            is_over = len(batch[0]) > 0
            start_idx += len(batch[0])
        else:
            batch = inputs[start_idx: start_idx + batch_size]
            is_over = len(batch) > 0
            start_idx += len(batch)
        if is_over > 0:
            yield batch
        else:
            break


def groupby_apply(values: torch.Tensor, keys: torch.Tensor, reduction: str = "mean"):
    """
    Groupby apply for torch tensors.
    Example: 
        Code:
            x = torch.FloatTensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
            g = torch.LongTensor([0, 0, 1, 1, 1])
            print(groupby_apply(x, g, 'mean'))
        Output:
            tensor([[1.5000, 1.5000],
                    [4.0000, 4.0000]])
    Args:
        values: values to aggregate - same size as keys
        keys: tensor of groups. 
        reduction: either "mean" or "sum"
    Returns:
        tensor with aggregated values
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    keys = keys.to(values.device)
    _, counts = keys.unique(return_counts=True)
    reduced = torch.stack([reduce(item, dim=0) for item in torch.split_with_sizes(values, tuple(counts))])
    return reduced


def plot_confusion(c_matrix, class_num, class_names=None,
                   norm_dec=2, cmap='Blues', info=''):
    """
    画出混淆矩阵。
    Args:
        c_matrix: 混淆矩阵
        class_num: 类别数量
        class_names: 各类名称，可选参数
        norm_dec: 标准化保留小数点位数
        cmap: 配色方案
        info: 显示在图像标题中的其他信息
    """
    title = 'Confusion matrix'

    data_size = c_matrix.sum()
    c_matrix = c_matrix.astype('int')

    fig = plt.figure()

    plt.imshow(c_matrix, interpolation='nearest', cmap=cmap)
    plt.title(f'{title} - ({data_size}) \n{info}')
    if class_names and len(class_names) == class_num:
        tick_marks = np.arange(class_num)
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names, rotation=0)

    thresh = c_matrix.max() / 2.
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        coeff = f'{c_matrix[i, j]}'
        plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                 color="yellow" if c_matrix[i, j] > thresh else "green")

    ax = fig.gca()
    ax.set_ylim(class_num-.5, -.5)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.grid(False)
    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig


def load_yaml(yaml_path, encoding='utf8'):
    """load yaml file.
    yaml file content:
        data_root: &root ./data                 # anchor define
        path: !path [*root, filename]           # use alias
        str: !str [*root, filename]
        items:
        - 1
        - 2
        - 3
        key: 
        key1: value1
        key2: value2
    load result:
        {
            'data_root': './data',
            'path': './data/filename', 
            'str': './datafilename', 
            'items': [1, 2, 3], 
            'key': {
                'key1':'value1',
                'key2': 'value2'
            }
        }
    """
    def path_fn(loader, node):
        items = loader.construct_sequence(node)
        return os.path.sep.join(items)

    def str_fn(loader, node):
        items = loader.construct_sequence(node)
        return ''.join([str(item) for item in items])

    def echo_fn(loader, node):
        seq = loader.construct_sequence(node)
        return seq[0]

    def sum_fn(loader, node):
        seq = loader.construct_sequence(node)
        return sum(seq)

    def mul_fn(loader, node):
        seq = loader.construct_sequence(node)
        return reduce(lambda x, y: x*y, seq)

    yaml.add_constructor('!path', path_fn)
    yaml.add_constructor('!join', path_fn)  # forward compatible
    yaml.add_constructor('!str', str_fn)
    yaml.add_constructor('!cat', str_fn)    # forward compatible
    yaml.add_constructor('!echo', echo_fn)
    yaml.add_constructor('!sum', sum_fn)
    yaml.add_constructor('!mul', mul_fn)

    with open(yaml_path, 'r', encoding=encoding) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return ddict(cfg)


class LinearCosineScheduler(_LRScheduler):
    """
    学习率调度器：先线性热身，然后余弦衰减。
    """
    def __init__(self, optimizer, max_val, warmup_steps, total_steps, min_val=None, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            max_val (float): 最大学习率
            warmup_steps (int): 热身次数
            total_steps (int): 最大调度次数
            min_val (float, optional): 最小学习率. Defaults to None.
            last_epoch (int, optional): Defaults to -1.
        """
        super().__init__(optimizer, last_epoch)
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        if min_val is None:
            min_val = max_val * (1 + np.cos((total_steps - warmup_steps + 1) * np.pi / (total_steps - self.warmup_steps + 2))) / 2
        self.min_val = min_val

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr = self.max_val * (step + 1) / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            lr = (self.max_val - self.min_val) * (1 + np.cos((step - self.warmup_steps + 1) * np.pi / (self.total_steps - self.warmup_steps + 1))) / 2 + self.min_val
        else:
            lr = self.min_val
        return [lr for _ in self.optimizer.param_groups]
