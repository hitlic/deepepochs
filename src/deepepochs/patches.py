"""
@author: liuchen
1. Patch是对一个或多个mini-batch运行结果的封装，用于batch指标计算、epoch指标计算。
2. Patch的重要方法：
    - __init__:  (metric, batch_size, name)      如果手动调用则参数无限制，如果在Trainer的参数则要求参数列表必须包含 (metric, batch_size)
    - forward:   ()                              用于计算或处理封装的数据，返回指标值或指标字典
    - load:      (batch_preds, batch_targets)    装入数据
    - add:       (other_obj)                     与另一对象相加
3. 内置Patch
    - PatchBase
        - patches.TensorPatch:        保存每个mini-batch的模型输出和标签                        可作为Trainer参数，也可用于step方法定制
        - patches.MeanPatch:          保存每个mini-batch的样本指标之和                          可作为Trainer参数，也可用于step方法定制
        - patches.ConfusionPatch:     保存每个mini-batch的混淆矩阵                             可作为Trainer参数，也可用于step方法定制
        - patches.ValuePatch:         保存每个mini-batch中每个样本某个值（如损失）之和            只能用于step定制
        - patches.ConfusionMetrics:   保存每个mini-batch的混淆矩阵，能计算多种基于混淆矩阵的指标    只能用于step定制
4. 使用Patch
    - 实例化对象
    - load方法装入数据（该方法返回对象自身）
    - forward方法计算指标
    - 利用sum对多个对象求和得到新对象，也可用 + 运算符
5. 定制Patch
    - 继承PatchBase，实现load方法、forward方法和add方法，参数要求参见 2.
"""
import abc
import traceback
from copy import copy, deepcopy
from typing import Literal

import torch
import torch.distributed as dist

from .loops import keyset
from .metrics import accuracy, confusion_matrix, fbeta, get_class_num, precision, recall


def patch_name(key, patch):
    name = getattr(patch, 'name', None)
    if name is None:
        return key
    else:
        return name


def run_patch_dict(patch_dict):
    """
    计算一个Patch字典的指标值（用于计算Batch指标）
    """
    return {patch_name(k, v): v() for k, v in patch_dict.items()}


def run_patch_dicts(patch_dicts):
    """
    计算Patch字典的列表的指标值（用于计算Epoch指标）
    """
    return run_patch_dict(_sum_patch_dicts(patch_dicts))


def _sum_patch_dicts(patch_dicts):
    if not patch_dicts:
        return {}
    return {key: sum(d[key] for d in patch_dicts if key in d)
            for key in keyset(patch_dicts)}


def _cpu_data(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_data(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_data(v) for v in value if not callable(v))
    return deepcopy(value)


def _tensor_device(value):
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return None
    for item in items:
        device = _tensor_device(item)
        if device is not None:
            return device
    return None


def _restore_data(value, prototype, device=None):
    """通信状态使用CPU，指标计算沿用本地原型的数据设备。"""
    local_device = _tensor_device(prototype)
    device = local_device if local_device is not None else device
    if isinstance(value, torch.Tensor):
        return value.to(device=device) if device is not None else value
    if isinstance(value, dict):
        return {key: _restore_data(item, prototype.get(key) if isinstance(prototype, dict) else None, device)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        originals = prototype if isinstance(prototype, (list, tuple)) else ()
        # 进程间批次数可不同，额外批次使用同一字段中已有Tensor的设备。
        return type(value)(_restore_data(item, originals[index] if index < len(originals) else None, device)
                           for index, item in enumerate(value))
    return value


def _statistic_metadata(value):
    if isinstance(value, torch.Tensor) and value.layout == torch.strided:
        return ('tensor', tuple(value.shape), value.dtype)
    if type(value) is float:
        return ('float', (), torch.float64)
    if type(value) is int:
        return ('int', (), torch.int64)
    if isinstance(value, dict):
        metadata = {key: _statistic_metadata(item) for key, item in value.items()}
        if all(item is not None for item in metadata.values()):
            return metadata
    return None


def _reduction_metadata(patch):
    # 只识别内置实现；子类的 add / forward 可以有不同的累计语义。
    if type(patch) in (ValuePatch, MeanPatch):
        fields = ('batch_size', 'batch_value')
        settings = None
    elif type(patch) in (ConfusionPatch, ConfusionMetrics):
        fields = ('confusion_matrix',)
        settings = (tuple(sorted(patch.metrics)), patch.average) if type(patch) is ConfusionMetrics else None
    else:
        return None
    state = {name: _statistic_metadata(getattr(patch, name)) for name in fields}
    if any(value is None for value in state.values()):
        return None
    return (type(patch).__name__, state, settings)


def _reduce_statistic(value, metadata, device):
    if isinstance(metadata, dict):
        return {key: _reduce_statistic(value[key] if value is not None else None, item, device)
                for key, item in metadata.items()}
    kind, shape, dtype = metadata
    tensor = torch.zeros(shape, dtype=dtype, device=device) if value is None else \
        torch.as_tensor(value, dtype=dtype, device=device).detach().clone()
    dist.all_reduce(tensor)
    if kind == 'tensor':
        return tensor.to(value.device) if value is not None else tensor
    return tensor.item()


def _patch_error(key, error):
    # traceback 对象不能跨进程传递，文本保留用户指标的原始调用位置。
    return key, type(error).__name__, str(error), traceback.format_exc()


def _check_gathered_states(gathered):
    for rank, states in enumerate(gathered):
        if not isinstance(states, dict):
            key, error_type, message, source_traceback = states
            raise RuntimeError(f'指标 {key!r} 状态聚合失败（rank {rank}，{error_type}）：{message}\n{source_traceback}')


class PatchAccumulator:
    """逐批累计本地 Patch，在任务结束时统一归约并计算指标。"""

    def __init__(self, context):
        self.context = context
        self.patches = {}
        self._error = None

    def add(self, patch_dict):
        if self._error is not None:
            return self
        try:
            for key, patch in patch_dict.items():
                self.patches[key] = self.patches[key] + patch if key in self.patches else 0 + patch
        except Exception as error:
            if not self.context.is_distributed:
                raise
            # 批次数可以不同：累计错误留到所有进程都进入 compute 时一起报告。
            self._error = _patch_error(key, error)
        return self

    def compute(self):
        context = self.context
        if not context.is_distributed:
            return run_patch_dict(self.patches)

        metadata = self._error
        if metadata is None:
            metadata = {}
            try:
                for key, patch in self.patches.items():
                    metadata[key] = _reduction_metadata(patch)
            except Exception as error:
                metadata = _patch_error(key, error)
        gathered_metadata = context.all_gather_object(metadata)
        _check_gathered_states(gathered_metadata)
        owners = {}
        for rank, states in enumerate(gathered_metadata):
            for key in states:
                owners.setdefault(key, rank)

        reductions = {}
        for key, owner in owners.items():
            spec = gathered_metadata[owner][key]
            if spec is not None and all(states[key] == spec for states in gathered_metadata if key in states):
                reductions[key] = spec

        # TensorPatch 和自定义 Patch 仍保留完整状态；函数及局部类留在 owner。
        fallback_keys = owners.keys() - reductions.keys()
        gathered_states = None
        if fallback_keys:
            try:
                states = {}
                for key, patch in self.patches.items():
                    if key in fallback_keys:
                        states[key] = {name: _cpu_data(value) for name, value in vars(patch).items()
                                       if not callable(value)}
            except Exception as error:
                states = _patch_error(key, error)
            gathered_states = context.all_gather_object(states)
            _check_gathered_states(gathered_states)

        # Gloo 使用 CPU；NCCL 归约前将 Python / CPU 统计放到当前 CUDA 设备。
        device = context.device if reductions and dist.get_backend() == 'nccl' else torch.device('cpu')
        metrics = {}
        for key, owner in owners.items():
            reduced = None
            if key in reductions:
                prototype = self.patches.get(key)
                reduced = {name: _reduce_statistic(getattr(prototype, name) if prototype is not None else None,
                                                 spec, device)
                           for name, spec in reductions[key][1].items()}
            payload = None
            if context.rank == owner:
                try:
                    prototype = self.patches[key]
                    if reduced is not None:
                        patch = copy(prototype)
                        patch.__dict__.update(reduced)
                    else:
                        patches = []
                        for states in gathered_states: # type: ignore
                            if key not in states:
                                continue
                            patch = copy(prototype)
                            patch.__dict__.update({name: _restore_data(value, getattr(prototype, name, None))
                                                   for name, value in states[key].items()})
                            patch.reusable = False
                            patches.append(patch)
                        patch = sum(patches)
                    payload = (patch_name(key, prototype), _cpu_data(patch()), None) # type: ignore
                except Exception as error:
                    payload = (None, None, _patch_error(key, error))
            name, value, error = context.broadcast_object(payload, src=owner)
            if error is not None:
                key, error_type, message, source_traceback = error
                raise RuntimeError(f'指标 {key!r} 聚合失败（rank {owner}，{error_type}）：{message}\n{source_traceback}')
            metrics[name] = value
        return metrics


def aggregate_patch_dicts(patch_dicts, context):
    accumulator = PatchAccumulator(context)
    for patch_dict in patch_dicts:
        accumulator.add(patch_dict)
    return accumulator.compute()


def _detach_data(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return type(value)({key: _detach_data(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_data(item) for item in value)
    return value


def _concat_data(values):
    first = values[0]
    if len(values) == 1 or first is None:
        return first
    if isinstance(first, dict):
        return {key: _concat_data([value[key] for value in values]) for key in first}
    if isinstance(first, (list, tuple)):
        return type(first)(_concat_data(list(items)) for items in zip(*values))
    if first.ndim == 0:
        return torch.stack(values)
    return torch.cat(values, dim=0)


class PatchBase(abc.ABC):
    """
    所有Patch对象的基类
    """
    def __init__(self, name=None):
        """
        Args:
            name: 显示在输出日志中的名称，当为空时使用指标函数的__name__属性
        """
        super().__init__()
        self.name = name
        self.reusable = False  # 当前对象是否可重复利用. 用于实现随意相加而不改变原始Patch数据, 见__add方法.

    def __add__(self, obj):
        return self.__add(obj)

    def __radd__(self, obj):
        return self.__add(obj)

    def __call__(self):
        return self.forward()

    @abc.abstractmethod
    def load(self, *values):
        """
        向当前Patch中装入数据
        """

    @abc.abstractmethod
    def forward(self):
        """
        基于当前Patch中保存的数据，计算一个结果（如指标值）并返回，被__call__方法自动调用。
        """

    @abc.abstractmethod
    def add(self, obj):
        """
        用于重载“+”运算符，将self和obj两个对象相加，得到一个新的对象。
        注意：在相加之前检查self和obj是否能够相加
        """

    def __add(self, obj):
        if obj == 0:                                        # 与 0 相加的情况
            if isinstance(self, TensorPatch):
                # TensorPatch在相加中会产生新对象，因而不必复制
                return self
            else:
                new_obj = deepcopy(self)
                new_obj.reusable = True
                return new_obj

        assert isinstance(obj, self.__class__), '相加的两个Patch的类型不一致！'

        # --- 下面的代码确保 add 方法一定由可重用对象调用
        if not self.reusable and not obj.reusable:          # 用 + 运算符相加的情况
            if isinstance(self, TensorPatch):
                new_obj = self
            else:
                new_obj = deepcopy(self)
                new_obj.reusable = True
            return new_obj.add(obj)
        elif self.reusable:                                 # 使用sum或者使用循环左加的情况
            return self.add(obj)
        else:
            return obj.add(self)                            # 使用循环右加的情况


def _weighted_statistic(value, batch_size):
    # Patch只保存指标统计，不保留训练计算图；加权乘法保持原设备和dtype。
    if isinstance(value, torch.Tensor):
        value = value.detach()
    return value * batch_size


class ValuePatch(PatchBase):
    def __init__(self, batch_size, name=None):
        """
        按样本数加权汇总batch均值，适用于平均损失及同类标量指标。
        load接受单个批均值或由多个批均值组成的字典；多个Patch相加后，
        forward返回参与汇总样本的加权均值。

        例如：
            vp1 = ValuePatch(batch_size=2).load(1.0)
            vp2 = ValuePatch(batch_size=3).load(3.0)
            combined = sum([vp1, vp2])
            combined()              # (2 * 1.0 + 3 * 3.0) / 5 = 2.2
            vp1()                   # 第一个batch的均值为1.0
            vp2()                   # 第二个batch的均值为3.0
        Args:
            batch_size:        mini-batch的大小
            name:              显示在输出日志中的名称
        """
        super().__init__(name)
        self.batch_size = batch_size
        self.batch_value = None

    def load(self, batch_mean_value):
        """
        Args:
            batch_mean_value:  一个mini-batch的平均值，例如平均损失；或者多个mini-batch平均值组成的字典。
        """
        if isinstance(batch_mean_value, dict):
            self.batch_value = {k: _weighted_statistic(v, self.batch_size) for k, v in batch_mean_value.items()}
        else:
            self.batch_value = _weighted_statistic(batch_mean_value, self.batch_size)
        return self

    def forward(self):
        if isinstance(self.batch_value, dict):
            return {k: v / self.batch_size for k, v in self.batch_value.items()}
        else:
            return self.batch_value / self.batch_size

    def add(self, obj):
        return add_patch_value(self, obj)


def add_patch_value(reuse_obj, obj):
    # PatchBase 的相加逻辑保证 reuse_obj 是可重用对象
    if isinstance(reuse_obj.batch_value, dict):
        assert isinstance(obj.batch_value, dict) and len(reuse_obj.batch_value) == len(obj.batch_value), '相加的两个Patch值不匹配！'
        keys = reuse_obj.batch_value.keys()
        keys_ = obj.batch_value.keys()
        assert len(set(keys).difference(set(keys_))) == 0, '相加的两个Patch值（字典）的key不一致！'
        reuse_obj.batch_value = {k: reuse_obj.batch_value[k]+obj.batch_value[k] for k in keys}
    else:
        reuse_obj.batch_value = reuse_obj.batch_value + obj.batch_value
    reuse_obj.batch_size = reuse_obj.batch_size + obj.batch_size
    return reuse_obj


class MeanPatch(PatchBase):
    def __init__(self, metric, batch_size, name=None):
        """
        用于累积多个mini-batch的指标值，计算Epoch的指标。
        Args:
            metric: 计算指标的函数（或其他适当的可调用对象），必须返回经过平均指标值。
            batch_size: 指定batch_size
            name: 显示在输出日志中的名称
        """
        super().__init__(name)
        assert callable(metric), '指标`metric`应当是一个可调用对象！'
        self.metric = metric
        self.batch_size = batch_size
        self.batch_value = None

    def load(self, batch_preds, batch_targets):
        """
        Args:
            batch_pres: 一个mini_batch的模型预测
            batch_targets: 一个mini_batch的标签
        """
        m_value = self.metric(batch_preds, batch_targets)
        if isinstance(m_value, dict):
            self.batch_value = {k: _weighted_statistic(v, self.batch_size) for k, v in m_value.items()}
        else:
            self.batch_value = _weighted_statistic(m_value, self.batch_size)
        return self

    def forward(self):
        if isinstance(self.batch_value, dict):
            return {k: v / self.batch_size for k, v in self.batch_value.items()}
        else:
            return self.batch_value / self.batch_size

    def add(self, obj):
        return add_patch_value(self, obj)


class TensorPatch(PatchBase):
    def __init__(self, metric, batch_size=None, name=None):
        """
        用于累积多个mini-batch的preds和targets，计算Epoch的指标。
        例如：
            batch 1的模型预测为preds1, 标签为targets1；
            batch 1的模型预测为preds2, 标签为targets2；
            m_fun 为指标计算函数；
            计算两个batch的指标：
                p1 = Patch(m_fun)
                p1.load(preds1, targets1)
                p2 = Patch(m_fun)
                p2.load(preds2, targets2)
                p = 0 + p1 + p2          # 两个Patch可直接相加，而且可与0相加
                p = sum([p1, p2])        # 可利用sum进行运算
                p1()  # batch 1上的指标值
                p2()  # batch 2上的指标值
                p()   # 两个batch上的指标值
        Args:
            metric:         计算指标的函数（或其他适当的可调用对象）
            batch_size:     指定batch_size（兼容性需要）
            name:           显示在输出日志中的名称
        """
        super().__init__(name)
        assert callable(metric), '指标`metric`应当是一个可调用对象！'
        self.reusable = True
        self.metric = metric
        self.batch_size = batch_size

    def load(self, batch_preds, batch_targets):
        """
        Args:
            batch_pres:     一个mini_batch的模型预测
            batch_targets:  一个mini_batch的标签（当指标计算不需要标签时为空值）
        """
        batch_preds = _detach_data(batch_preds)
        batch_targets = _detach_data(batch_targets)
        self.batch_preds = [batch_preds] if isinstance(batch_preds, (list, tuple)) else [[batch_preds]]
        self.batch_targets = None if batch_targets is None else \
            ([batch_targets] if isinstance(batch_targets, (list, tuple)) else [[batch_targets]])
        return self

    def forward(self):
        try:
            # 合并各batch中的模型输出
            preds = [_concat_data(list(bpreds)) for bpreds in zip(*self.batch_preds)]
            # 合并各batch的学习目标（标签）
            targets = None if self.batch_targets is None else \
                    [_concat_data(list(btargets)) for btargets in zip(*self.batch_targets)]
        # 如果concat失败则放弃concat，以应对复杂的模型输出；而且要考虑到在GNN中batchsize为1的情况。
        except (RuntimeError, TypeError):
            preds = self.batch_preds
            targets = self.batch_targets

        preds = preds[0] if len(preds) == 1 else preds
        targets = targets[0] if targets is not None and len(targets) == 1 else targets
        return self.metric(preds, targets)

    def add(self, obj):
        assert self.metric is obj.metric, '相加的两个Patch的`metric`不一致'
        new_preds = self.batch_preds + obj.batch_preds
        assert (self.batch_targets is None) == (obj.batch_targets is None), '相加的两个Patch的`batch_targets`其中一个为None！'
        if self.batch_targets is not None:
            new_targets = self.batch_targets + obj.batch_targets
        else:
            new_targets = None
        new_patch = self.__class__(self.metric, self.batch_size, self.name)
        new_patch.batch_preds = new_preds
        new_patch.batch_targets = new_targets
        return new_patch


class ConfusionPatch(PatchBase):
    def __init__(self, metric, batch_size=None, name=None):
        """
        累积计算混淆矩阵的Patch。
        Args:
            metric:        以混淆矩阵为输入的指标
            batch_preds:    模型预测
            batch_targets:  标签
            batch_size:     指定batch_size（兼容性需要）
            name:           显示在输出日志中的名称
        """
        super().__init__(name)
        self.batch_size = batch_size
        self.metric = metric

    def load(self, batch_preds, batch_targets):
        num_classes = get_class_num(batch_preds, batch_targets)
        self.num_classes = num_classes
        self.confusion_matrix = confusion_matrix(batch_preds, batch_targets, num_classes)
        return self

    def forward(self):
        return self.metric(self.confusion_matrix)

    def add(self, obj):
        # PatchBase 的相加逻辑保证 self 是可重用对象
        assert self.confusion_matrix.shape == obj.confusion_matrix.shape, '相加的两个Patch中数据的类别数量不相等！'
        self.confusion_matrix = self.confusion_matrix + obj.confusion_matrix
        return self


class ConfusionMetrics(PatchBase):
    def __init__(self,
                 metrics=('accuracy', 'precision', 'recall', 'f1', 'fbeta'),
                 average: Literal['micro', 'macro', 'weighted']='micro', beta=1.0, name='C.'):
        """
        能够累积计算基于混淆矩阵的指标，包括'accuracy', 'precision', 'recall', 'f1', 'fbeta'等。可在定制的steps方法中使用。
        Args:
            batch_preds:    模型预测
            batch_targets:  标签
            metrics:        需计算的指标，'accuracy', 'precision', 'recall', 'f1', 'fbeta'中的一个或多个
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

    def load(self, batch_preds, batch_targets):
        num_classes = get_class_num(batch_preds, batch_targets)
        self.num_classes = num_classes
        self.confusion_matrix = confusion_matrix(batch_preds, batch_targets, num_classes)
        return self

    def forward(self):
        return {self.metric2name[m]: getattr(self, m)() for m in self.metrics}

    def add(self, obj):
        assert self.confusion_matrix.shape == obj.confusion_matrix.shape, '相加的两个Patch中数据的类别数量不相等！'
        assert set(self.metrics) == set(obj.metrics), '相加的两个Patch的`metrics`不一致!'
        assert self.average == obj.average, '相加的两个Patch的`average`不一致!'

        # PatchBase 的相加逻辑保证 self 是可重用对象
        self.confusion_matrix = self.confusion_matrix + obj.confusion_matrix
        return self

    def accuracy(self):
        return accuracy(conf_mat=self.confusion_matrix)

    def precision(self):
        return precision(average=self.average, conf_mat=self.confusion_matrix)

    def recall(self):
        return recall(average=self.average, conf_mat=self.confusion_matrix)

    def fbeta(self):
        return fbeta(average=self.average, beta=self.beta, conf_mat=self.confusion_matrix)

    def f1(self):
        return fbeta(average=self.average, beta=1, conf_mat=self.confusion_matrix)
