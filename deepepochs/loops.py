"""
@author: hitlic
DeepEpochs is a simple Pytorch deep learning model training tool(see https://github.com/hitlic/deepepochs).
"""
import os
from os import path as osp
import abc
import torch
import numpy as np
from typing import Literal


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


class TensorTuple(tuple):
    """
    list of tensors
    """
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
        return str(to_numpy(v).round(round_to)).ljust(round_to+3)
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


class PatchBase(abc.ABC):
    """
        所有Patch对象的基类。
        Patch对象对一个mini-batch的模型输出数据和数据的处理方法进行了封装，用于方便地计算多个mini-batch或epoch的
    累积结果。主要用于平均损失和累积计算（ValuePatch）和指标的累积计算（TensorPatch）。
        TensorPatch封装了指标函数、模型预测输出和标签，计算和空间效率较低，但能够用于计算任意指标。MeanPack封装了指
    标函数、当前batch的指标值和batch_size，计算和存储效率较高，但不能用于无法进行简单累积求均值的指标（基于混淆矩阵的
    指标）。可以通过定义新的Patch类型实现更复杂的指标计算。
        参见ValuePatch、TensorPatch、MeanPatch类，以及trainer.Trainer的train_step方法和evaluate_step方法。
    """
    def __init__(self, name=None):
        """
        Args:
            name: 显示在输出日志中的名称
        """
        super().__init__()
        self.name = name

    def __add__(self, obj):
        return self.__add(obj)

    def __radd__(self, obj):
        return self.__add(obj)

    def __call__(self):
        return self.forward()

    @abc.abstractmethod
    def forward(self):
        """
        基于当前Patch中保存的数据，计算一个结果（如指标值）并返回，被__call__方法自动调用。
        """

    def __add(self, obj):
        if obj == 0:
            return self
        assert isinstance(obj, self.__class__), '相加的两个Patch的类型不一致！'
        return self.add(obj)

    @abc.abstractmethod
    def add(self, obj):
        """
        用于重载“+”运算符，将self和obj两个对象相加，得到一个新的对象。
        注意：在相加之前检查self和obj是否能够相加
        """


class ValuePatch(PatchBase):
    def __init__(self, batch_mean_value, batch_size, name=None):
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
            name: 显示在输出日志中的名称
        """
        super().__init__(name)
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
        return add_patch_value(self, obj)


def add_patch_value(self_obj, obj):
    if isinstance(self_obj.batch_value, dict):
        assert isinstance(obj.batch_value, dict) and len(self_obj.batch_value) == len(obj.batch_value), '相加的两个Patch值不匹配！'
        keys = self_obj.batch_value.keys()
        keys_ = obj.batch_value.keys()
        assert len(set(keys).difference(set(keys_))) == 0, '相加的两个Patch值（字典）的key不一致！'
        self_obj.batch_value = {k: self_obj.batch_value[k]+obj.batch_value[k] for k in keys}
    else:
        self_obj.batch_value += obj.batch_value
    self_obj.batch_size += obj.batch_size
    return self_obj


class TensorPatch(PatchBase):
    def __init__(self, metric, batch_preds, batch_targets=None, name=None):
        """
        用于累积多个mini-batch的preds和targets，计算Epoch的指标。
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
            name: 显示在输出日志中的名称
        """
        super().__init__(name)
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
        assert self.metric is obj.metric, '相加的两个Patch的`metric`不一致'
        new_preds = self.batch_preds + obj.batch_preds
        if self.batch_targets != None:
            assert obj.batch_targets is not None, '相加的两个Patch的`batch_targets`其中一个为None！'
            new_targets = self.batch_targets + obj.batch_targets
        else:
            new_targets = None
        return self.__class__(self.metric, new_preds, new_targets, self.name)


class MeanPatch(PatchBase):
    def __init__(self, metric, batch_preds, batch_targets=None, name=None):
        """
        用于累积多个mini-batch的指标值，计算Epoch的指标。
        Args:
            metric: 计算指标的函数（或其他适当的可调用对象），必须返回经过平均指标值。
            batch_pres: 一个mini_batch的模型预测
            batch_targets: 一个mini_batch的标签（当指标计算不需要标签时为空值）
            name: 显示在输出日志中的名称
        """
        super().__init__(name)
        assert callable(metric), '指标`metric`应当是一个可调用对象！'
        self.metric = metric
        self.batch_size = len(batch_preds)
        m_value = metric(batch_preds, batch_targets)
        if isinstance(m_value, dict):
            self.batch_value = {k: v * self.batch_size for k, v in m_value.items()}
        else:
            self.batch_value = m_value * self.batch_size

    def forward(self):
        if isinstance(self.batch_value, dict):
            return {k: v / self.batch_size for k, v in self.batch_value.items()}
        else:
            return self.batch_value / self.batch_size

    def add(self, obj):
        assert self.metric is obj.metric, '相加的两个Patch的`metric`不一致'
        return add_patch_value(self, obj)


class ConfusionPatch(PatchBase):
    def __init__(self, batch_preds, batch_targets,
                 metrics=('accuracy', 'precision', 'recall', 'f1', 'fbeta'),
                 average: Literal['micro', 'macro', 'weighted']='micro', beta=1.0, name='C.'):
        """
        能够累积计算基于混淆矩阵的指标，包括'accuracy', 'precision', 'recall', 'f1', 'fbeta'等。
        Args:
            batch_preds:    模型预测
            batch_targets:  标签
            metrics:        需计算的标签，'accuracy', 'precision', 'recall', 'f1', 'fbeta'中的一个或多个
            average:        多分类下的平均方式'micro', 'macro', 'weighted'之一
            beta:           F_beta中的beta
            name:           显示在输出日志中的名称
        """
        super().__init__(name)
        if isinstance(metrics, str):
            metrics = [metrics]

        assert set(metrics) <= set(['accuracy', 'precision', 'recall', 'f1', 'fbeta']),\
                "未知`metrics`！可取值为{'accuracy', 'precision', 'recall', 'f1', 'fbeta'}的子集！"
        assert average in ['micro', 'macro', 'weighted'], "`average`取值为['micro', 'macro', 'weighted']之一！"
        self.metric2name = {'accuracy': 'acc', 'recall': 'r', 'precision': 'p', 'f1': 'f1', 'fbeta': 'fb'}

        if 'fbeta' in metrics:
            assert beta > 0, 'F_beta中的beta必须大于0！'
            self.beta = beta

        self.metrics = metrics
        self.average = average

        if batch_preds.shape[1] == 1:
            num_classes = None
        else:
            num_classes = batch_preds.shape[1]
        self.num_classes = num_classes
        self.confusion_matrix = self._confusion_matrix(batch_preds, batch_targets)

    def _confusion_matrix(self, preds, targets):
        preds = preds.argmax(dim=1)
        cm = torch.zeros([self.num_classes, self.num_classes], dtype=preds.dtype, device=preds.device)
        one = torch.tensor([1], dtype=preds.dtype, device=preds.device)
        return cm.index_put_((preds, targets), one, accumulate=True)


    def forward(self):
        c_mats = self.get_c_mats()
        weights = [mat.TP+mat.FN for mat in c_mats]
        w_sum = sum(weights)
        weights = [w/w_sum for w in weights]
        return {self.metric2name[m]: getattr(self, m)(c_mats, weights) for m in self.metrics}

    def add(self, obj):
        assert self.confusion_matrix.shape == obj.confusion_matrix.shape, '相加的两个Patch中数据的类别数量不相等！'
        assert set(self.metrics) == set(obj.metrics), '相加的两个Patch的`metrics`不一致!'
        assert self.average == obj.average, '相加的两个Patch的`average`不一致!'
        self.confusion_matrix += obj.confusion_matrix
        return self

    def get_c_mats(self):
        if self.confusion_matrix.shape[0] == 2:
            c_mat = ddict({
                'TP': self.confusion_matrix[0][0],
                'FN': self.confusion_matrix[0][1],
                'FP': self.confusion_matrix[1][0],
                'TN': self.confusion_matrix[1][1]
            })
            return [c_mat]
        else:
            return [ddict(self.get_cmat_i(i)) for i in range(self.confusion_matrix.shape[0])]

    def get_cmat_i(self, c_id):
        TP = self.confusion_matrix[c_id, c_id]
        FN = self.confusion_matrix[c_id].sum() - TP
        FP = self.confusion_matrix[:, c_id].sum() - TP
        TN = self.confusion_matrix.sum() - TP - FN - FP
        return ddict({'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN})

    def accuracy(self, _1, _2):
        return sum(self.confusion_matrix[i, i] for i in range(self.num_classes))/self.confusion_matrix.sum()

    def precision(self, c_mats, weights):
        if self.average == 'micro':
            return precision_fn(sum_dicts(c_mats))
        elif self.average == 'macro':
            return sum(precision_fn(mat) for mat in c_mats)/self.num_classes
        else:
            ps = [precision_fn(mat) for mat in c_mats]
            return sum(p*w for p, w in zip(ps, weights))

    def recall(self, c_mats, weights):
        if self.average == 'micro':
            return recall_fn(sum_dicts(c_mats))
        elif self.average == 'macro':
            return sum(recall_fn(mat) for mat in c_mats)/self.num_classes
        else:
            ps = [recall_fn(mat) for mat in c_mats]
            return sum(p*w for p, w in zip(ps, weights))

    def fbeta(self, c_mats, weights):
        return self._fbeta(c_mats, weights, self.beta)

    def _fbeta(self, c_mats, weights, beta):
        if self.average == 'micro':
            return fbeta_fn(sum_dicts(c_mats), beta)
        elif self.average == 'macro':
            return sum(fbeta_fn(mat, beta) for mat in c_mats)/self.num_classes
        else:
            ps = [fbeta_fn(mat, beta) for mat in c_mats]
            return sum(p*w for p, w in zip(ps, weights))

    def f1(self, c_mats, weights):
        return self._fbeta(c_mats, weights, 1)

def precision_fn(c_mat):
    if c_mat.TP + c_mat.FP == 0:
        return 0
    return c_mat.TP/(c_mat.TP + c_mat.FP)

def recall_fn(c_mat):
    if c_mat.TP + c_mat.FP == 0:
        return 0
    return c_mat.TP/(c_mat.TP + c_mat.FN)

def fbeta_fn(c_mat, beta):
    p = precision_fn(c_mat)
    r = recall_fn(c_mat)
    if p + r == 0:
        return 0
    return (1 + beta**2) * (p*r)/(beta**2 * p + r)


def run_patch_dict(patch_dict):
    """
    计算一个Patch字典的指标值（计算Batch指标）
    """
    return {patch_name(k, v): v() for k, v in patch_dict.items()}


def run_patch_dicts(patch_dicts):
    """
    计算Patch字典的列表的指标值（计算Epoch指标）
    """
    if len(patch_dicts) == 0:
        return None
    return {patch_name(k, patch_dicts[0][k]): sum(dic[k] for dic in patch_dicts)() for k in keyset(patch_dicts)}


def patch_name(k, patch):
    name = getattr(patch, 'name', None)
    if name is None:
        return k
    else:
        return name


def default_loss(preds, targets):
    """默认损失函数，直接返回模型预测结果，适用于模型直接返回损失值的情况。"""
    return preds


class StopLoopException(Exception):
    pass


class LoopException(Exception):
    pass


class Optimizer:
    def __init__(self, opt, scheduler=None, sched_on='epoch', sched_with_loss=False):
        """
        优化器组合，对优化器和学习率调度器进行统一管理。
        Args:
            opt:             torch.optim.*
            scheduler:       torch.optim.lr_scheduler.*
            sched_on:        学习率调整是每个epoch还是每个step
            sched_with_loss: scheduler.step方法是否需要损失作为参数（例如ReduceLROnPlateau）
        """
        self.opt = opt
        self.scheduler = scheduler
        assert sched_on in ['step', 'epoch'], '`sched_on`取值为"step"或"epoch"!'
        self.sched_on = sched_on
        self.sched_with_loss = sched_with_loss

    def zero_grad(self):
        self.opt.zero_grad()

    def get_last_lr(self):
        return self.scheduler.get_last_lr() if self.scheduler is not None else None

    def step(self, at='step', loss=None):
        if at == 'step':
            self.opt.step()
            if self.sched_on == 'step':
                self.sched_step(loss)
        elif at == 'epoch':
            if self.sched_on == 'epoch':
                self.sched_step(loss)
        else:
            raise ValueError('Optimizer.step方法的`at`参数取值为"step"或"epoch"')

    def sched_step(self, loss):
        if self.scheduler is not None:
            if self.sched_with_loss:
                assert loss is not None, "学习率调度要求损失作为参数，但`train_step`和`evaluate_step`都没有返回`loss`！"
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

    def state_dict(self):
        if self.scheduler is None:
            return self.opt.state_dict()
        return self.opt.state_dict(), self.scheduler.state_dict()

    def load_state_dict(self, opt_state, sched_state=None):
        self.opt.load_state_dict(opt_state)
        if sched_state is not None:
            self.scheduler.load_state_dict(opt_state)

    def get_current_lr(self):
        for param_group in self.opt.param_groups:
            return param_group['lr']


class Optimizers(list):
    """
    用于管理多个优化器组合（Optimizer），对多个优化器提供支持。
    """
    def zero_grad(self):
        for opt in self:
            opt.zero_grad()

    def get_last_lr(self):
        return [opt.get_last_lr() for opt in self]

    def step(self, at='step', loss=None):
        for opt in self:
            opt.step(at, loss)

    def state_dict(self):
        return [opt.state_dict for opt in self]

    def load_state_dict(self, opt_states, sched_states=None):
        if sched_states is None:
            for opt, opt_state in zip(self, opt_states):
                opt.load_state_dict(opt_state)
        else:
            for opt, opt_state, sched_state in zip(self, opt_states, sched_states):
                opt.load_state_dict(opt_state, sched_state)

    def get_current_lr(self):
        return [opt.get_current_lr() for opt in self]
