"""
本模块代码尽量不要变动。
"""
import os
from os import path as osp
import abc
from copy import deepcopy
from collections import defaultdict
import torch
from torch.optim import Adam
import numpy as np
from typing import List, Callable


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


def log_batch(metrics, epoch_idx, epochs, batch_idx, batchs, stage):
    """输出batch指标值"""
    batch_info = info(metrics)
    print_out(f'E{epoch_idx:>4}/{epochs:<4} B{batch_idx:>4}/{batchs:<5} {stage}> {batch_info}', end='')


def log_epoch(stages_metrics, epoch_idx, epochs):
    """输出epoch指标值"""
    train_metrics = stages_metrics.get('train')
    val_metrics = stages_metrics.get('val')
    test_metrics = stages_metrics.get('test')
    if train_metrics is not None:
        train_info = info(train_metrics)
        val_info = ''
        if val_metrics is not None:
            val_info = info(val_metrics)
            val_info = '  VAL> ' + val_info
        print_out(f'E{epoch_idx:>4}/{epochs:<4} TRAIN> {train_info}{val_info}')
    elif test_metrics is not None:
        test_info = info(test_metrics)
        print_out(f'E{epoch_idx:>4}/{epochs:<4} TEST> {test_info}')
    else:
        raise ValueError("log_epoch 参数错误!")


def info(m_dict):
    return ' '.join([f'{k}: {str(to_numpy(v).round(6)):<8}' for k, v in m_dict.items()])


def print_out(content, end='\r\n'):
    print('\x1b[1K\r', end='')
    print(content, flush=True, end=end)


def concat_dicts(dicts):
    return {k: [to_numpy(d.get(k, 0)) for d in dicts] for k in keyset(dicts)}


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
            return d
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
            print(f'The path `{path}` is created!')
            os.makedirs(path)
        else:
            raise ValueError(f'Path "{path}" does not exists!')


class Checker:
    def __init__(self, monitor, mode='max', patience=None, path='./logs/checkpoint'):
        """
        实现了checkpoint和earlystop。
        Args:
            monitor: metric name in validation stage
            mode: max or min
            patience: early stopping patience
            path: path to save the best checkpoint
        """
        self.model = None      # 模型
        self.monitor = monitor
        self.mode = mode
        self.patience = patience

        assert mode in ['min', 'max']
        if mode == 'max':
            self.best_value = -100000000.0
        else:
            self.best_value = 100000000.0

        check_path(path)
        self.path = osp.join(path, 'model.ckpt')

        self.worse_times = 0

    def check(self, metrics):
        value = metrics[self.monitor]
        if self.mode == 'max':
            if  value > self.best_value:
                self.best_value = value
                self.save_model()
                self.worse_times = 0
            else:
                self.worse_times += 1
        else:
            if value < self.best_value:
                self.best_value = value
                self.save_model()
                self.worse_times = 0
            else:
                self.worse_times += 1
        if self.patience is not None and self.worse_times >= self.patience:
            return False
        return True

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)

    def load_best(self):
        self.model.load_state_dict(torch.load(self.path))


class PatchBase(abc.ABC):
    """
        所有Patch对象的基类。
        Patch对象对一个mini-batch的模型输出数据和数据的处理方法进行了封装，用于方便地计算多个mini-batch或epoch
    的累积结果。主要用于平均损失和累积计算（ValuePack）和指标的累积计算（TensorPack）。
        TensorPack封装了指标函数、模型预测输出和标签，可能会占用较多存储空间。可以通过定义新的Patch类型实现更高效
    的指标计算，例如基于混淆矩阵的指标等。
        参见ValuePack和TensorPack代码和文档，以及trainer.Trainer的train_step方法和evaluate_step方法。
    """
    def __add__(self, obj):
        return self.add(obj)

    def __radd__(self, obj):
        return self.add(obj)

    def __call__(self):
        return self.forward()

    @abc.abstractmethod
    def forward(self):
        """
        基于当前Patch中保存的数据，计算一个结果（如指标值）并返回，被__call__方法自动调用。
        """

    @abc.abstractmethod
    def add(self, obj):
        """
        用于重载“+”运算符，将self和obj两个对象相加，得到一个新的对象。
        注意1：如果obj为0则返回self
        注意2：在相加之前检查self和obj是否能够相加
        """
        if obj == 0:
            return self
        return None


class ValuePatch(PatchBase):
    def __init__(self, batch_mean_value, batch_size):
        """
        主要用于根据mini-batch平均损失得到epoch平均损失（也可用于与损失相似的数值的累积平均计算），支持字典值的累积平均计算。
        例如：
            batch 1的平均损失为 loss1, 批量大小为 bsize1；
            batch 2的平均损失为 loss2, 批量大小为 bsize3；
            计算两个batch的平均损失：
                vp1 = ValuePatch(loss1, bsize1)
                vp2 = ValuePatch(loss2, bsize2)
                vp = 0 + vp1 + vp2      # 两个Patch直接相加，而且可以与0相加
                vp = sum([vp1, vp2])    # 可利用sum进行运算
                vp1()                            # batch 1上的平均损失
                vp2()                            # batch 2上的平均扣抽
                vp()                    # 两个mini-batch的平均损失
        Args:
            batch_mean_value: 一个mini-batch的平均值，例如平均损失；或者多个mini-batch平均值组成的字典。
            batch_size: mini-batch的大小
        """
        super().__init__()
        if isinstance(batch_mean_value, dict):
            self.batch_value = {k: v * batch_size for k, v in batch_mean_value.items()}
        else:
            self.batch_value = batch_mean_value * batch_size
        self.batch_size = batch_size

    def forward(self):
        if isinstance(self.batch_value, dict):
            return {k: v / self.batch_size for k, v in self.batch_value.items()}
        else:
            return self.batch_value / self.batch_size

    def add(self, obj):
        if obj == 0:
            return self
        assert isinstance(obj, self.__class__), '相加的两个Patch的类型不一致！'
        new_obj = deepcopy(self)
        if isinstance(self.batch_value, dict):
            assert isinstance(obj.batch_value, dict) and len(self.batch_value) == len(obj.batch_value), '相加的两个Patch值不匹配！'
            keys = self.batch_value.keys()
            keys_ = obj.batch_value.keys()
            assert len(set(keys).difference(set(keys_))) == 0, '相加的两个Patch值（字典）的key不一致！'
            new_obj.batch_value = {k: new_obj.batch_value[k]+obj.batch_value[k] for k in keys}
        else:
            new_obj.batch_value += obj.batch_value
        new_obj.batch_size += obj.batch_size
        return new_obj


class TensorPatch(PatchBase):
    def __init__(self, metric, batch_preds, batch_targets=None):
        """
        用于类积多个mini-batch的preds和targets，计算Epoch的指标。
        例如：
            batch 1的模型预测为preds1, 标签为targets1；
            batch 1的模型预测为preds2, 标签为targets2；
            m_fun 为指标计算函数；
            计算两个batch的指标：
                p1 = Patch(m_fun, preds1, targets1)
                p2 = Patch(m_fun, preds2, targets2)
                p = 0 + p1 + p2          # 两个Patch可直接相加，而且可与0相加
                p = sum([p1, p2])        # 可利用sum进行运算
                p1()  # batch 1上的指标值
                p2()  # batch 2上的指标值
                p()   # 两个batch上的指标值
        Args:
            metric: 计算指标的函数（或其他适当的可调用对象）
            batch_pres: 一个mini_batch的模型预测
            batch_targets: 一个mini_batch的标签（当指标计算不需要标签时为空值）
        """
        super().__init__()
        assert callable(metric), '指标`metric`应当是一个可调用对象！'
        self.metric = metric
        self.batch_preds = list(batch_preds) if isinstance(batch_preds, (list, tuple)) else [batch_preds]
        if batch_targets is None:
            self.batch_targets = None
        else:
            self.batch_targets = list(batch_targets) if isinstance(batch_targets, (list, tuple)) else [batch_targets]

        self.concat = torch.concat if isinstance(self.batch_preds[0], torch.Tensor) else np.concatenate

    def forward(self):
        preds = self.concat(self.batch_preds, 0)
        targets = None if self.batch_targets is None else self.concat(self.batch_targets, 0)
        return self.metric(preds, targets)

    def add(self, obj):
        if obj == 0:
            return self
        assert isinstance(obj, self.__class__), '相加的两个Patch的类型不一致！'
        assert self.metric is obj.metric, '相加的两个Patch的`metric`不一致'
        new_preds = self.batch_preds + obj.batch_preds
        if self.batch_targets != None:
            assert obj.batch_targets is not None, '相加的两个Patch的`batch_targets`其中一个为None！'
            new_targets = self.batch_targets + obj.batch_targets
        else:
            new_targets = None
        return self.__class__(self.metric, new_preds, new_targets)


def exec_dict(patch_dict):
    """
    计算一个Patch字典的指标值（计算Batch指标）
    """
    return {k: v() for k, v in patch_dict.items()}


def exec_dicts(patch_dicts):
    """
    计算Patch字典的列表的指标值（计算Epoch指标）
    """
    if len(patch_dicts) == 0:
        return None
    return {k: sum(dic[k] for dic in patch_dicts)() for k in keyset(patch_dicts)}


def default_loss(preds, targets):
    """默认损失函数，直接返回模型预测结果，适用于模型直接返回损失值的情况。"""
    return preds


class TrainerBase:
    def __init__(self, model:torch.nn.Module, loss:Callable=None, opt:torch.optim.Optimizer=None, epochs:int=1000,
                 metrics: List[Callable]=None, device=None, val_freq:int=1, checker:Checker=None):
        """
        Args:
            model:      Pytorch模型
            loss:       损失函数
            opt:        优化器
            epochs:     迭代次数
            metrics:    指标
            device:     cpu或cuda
            val_freq:   验证频率
            checker:    Checker类的对象，实现了checkpoint和early stopping
        """
        # 配置损失函数
        if loss is None:
            self.loss = default_loss
        else:
            self.loss = loss
        # 配置优化器
        if opt is None:
            self.opt = Adam(model.parameters(), lr=0.001)
        else:
            self.opt = opt
        self.epochs = epochs        # 迭代次数
        self.metrics = metrics      # 指标
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.val_freq = val_freq    # 验证频率

        # 配置checkpoint和early stop
        self.checker = checker
        if checker is not None:
            self.checker.model = model

    def fit(self, train_dl, val_dl=None):
        """该方法尽量不要变动"""
        progress = defaultdict(list)   # 保存各epoch的指标值
        try:
            for epoch_idx in range(self.epochs):
                # training
                self.model.train()
                train_metrics = []
                batchs = len(train_dl)

                for batch_idx, (batch_x, batch_y) in enumerate(train_dl):
                    train_ms = self.train_step(batch_x.to(self.device), batch_y.to(self.device))
                    train_metrics.append(train_ms)
                    with torch.no_grad():
                        # 计算当前batch的指标并输出
                        log_batch(flatten_dict(exec_dict(train_ms), sep=''), epoch_idx+1, self.epochs, batch_idx+1, batchs, 'TRAIN')
                with torch.no_grad():
                    # 计算当前epoch的指标
                    train_metrics = flatten_dict(exec_dicts(train_metrics), sep='')
                progress['train'].append(train_metrics)

                # validation
                if val_dl is not None and (epoch_idx + 1) % self.val_freq == 0:
                    self.model.eval()
                    val_metrics = []
                    batchs = len(val_dl)
                    with torch.no_grad():
                        for batch_idx, (batch_x, batch_y) in enumerate(val_dl):
                            val_ms = self.evaluate_step(batch_x.to(self.device), batch_y.to(self.device))
                            val_metrics.append(val_ms)
                            # 计算当前batch的指标并输出
                            log_batch(flatten_dict(exec_dict(val_ms), sep=''), epoch_idx+1, self.epochs, batch_idx+1, batchs, 'VAL')
                        # 计算当前epoch的指标
                        val_metrics = flatten_dict(exec_dicts(val_metrics), sep='')
                        progress['val'].append(val_metrics)
                    # 输出当前epoch的训练和验证结果
                    log_epoch({'train': train_metrics, 'val': val_metrics}, epoch_idx+1, self.epochs)
                    # 检查是否需要保存checkpoint、是否满足早停条件
                    if self.checker is not None and not self.checker.check(val_metrics):
                        print('Early stopping triggered!')
                        break
                else:
                    log_epoch({'train': train_metrics}, epoch_idx+1, self.epochs)

        except KeyboardInterrupt:
            print('\nStop trainning manually!')
        return {k: concat_dicts(v) for k, v in progress.items()}


    def test(self, test_dl):
        print('-'*30)
        if self.checker is not None:
            print('loading best model ...')
            self.checker.load_best()  # 加载最优模型
        # testing
        self.model.eval()
        test_metrics = []
        batchs = len(test_dl)
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(test_dl):
                test_ms = self.evaluate_step(batch_x.to(self.device), batch_y.to(self.device))
                test_metrics.append(test_ms)
                # 计算当前batch的指标并输出
                log_batch(flatten_dict(exec_dict(test_ms), sep=''), 1, 1, batch_idx+1, batchs, 'TEST')
            # 计算当前epoch的指标
            test_metrics = flatten_dict(exec_dicts(test_metrics), sep='')
        log_epoch({'test': test_metrics}, 1, 1)
        return to_numpy(test_metrics)

    def train_step(self, batch_x, batch_y):
        """
        TODO: 非常规训练可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据的ValuePatch或者Patch。
        """
        raise NotImplementedError("Trainer.train_step 方法未实现！")

    def evaluate_step(self, batch_x, batch_y):
        """
        TODO: 非常规验证或测试可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据的ValuePatch或者Patch。
        """
        raise NotImplementedError("Trainer.evaluate_step 方法未实现！")
