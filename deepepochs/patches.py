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
    - 继承MetricPatch，实现load方法、forward方法和add方法，参数要求参见 2.
"""
import abc
from typing import Literal
import torch
from .metrics import confusion_matrix, accuracy, precision, recall, fbeta
from .loops import keyset, clone_value
from copy import deepcopy


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
    if len(patch_dicts) == 0:
        return None
    return {patch_name(k, patch_dicts[0][k]): sum(dic[k] for dic in patch_dicts if dic)() for k in keyset(patch_dicts)}


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
        if obj == 0:
            # 使用sum求和时，第一步与0相加，此时复制出一份新的对象一直使用，不影响原始数据
            #    如是TensorPatch则返回自身，因为它只是对数据的引用列表，不涉及大量的数据变动
            return deepcopy(self) if not isinstance(self, TensorPatch) else self
        assert isinstance(obj, self.__class__), '相加的两个Patch的类型不一致！'
        return self.add(obj)


class ValuePatch(PatchBase):
    def __init__(self, batch_size, name=None):
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
                vp1()                   # batch 1上的平均损失
                vp2()                   # batch 2上的平均扣抽
                vp()                    # 两个mini-batch的平均损失
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
            self.batch_value = {k: v * self.batch_size for k, v in batch_mean_value.items()}
        else:
            self.batch_value = batch_mean_value * self.batch_size
        return self

    def forward(self):
        if isinstance(self.batch_value, dict):
            return {k: v / self.batch_size for k, v in self.batch_value.items()}
        else:
            return self.batch_value / self.batch_size

    def add(self, obj):
        return add_patch_value(self, obj)

    def clone(self):
        """本方法已废弃"""
        new_obj = self.__class__(self.batch_size, self.name)
        new_obj.batch_value = clone_value(self.batch_value)
        return new_obj


def add_patch_value(self_obj, obj):
    # new_obj = self_obj.clone()
    new_obj = self_obj                  # 由于93行使用deepcopy复制了一个新的对象，因此没必要每次新建对象
    if isinstance(new_obj.batch_value, dict):
        assert isinstance(obj.batch_value, dict) and len(new_obj.batch_value) == len(obj.batch_value), '相加的两个Patch值不匹配！'
        keys = new_obj.batch_value.keys()
        keys_ = obj.batch_value.keys()
        assert len(set(keys).difference(set(keys_))) == 0, '相加的两个Patch值（字典）的key不一致！'
        new_obj.batch_value = {k: new_obj.batch_value[k]+obj.batch_value[k] for k in keys}
    else:
        new_obj.batch_value += obj.batch_value
    new_obj.batch_size += obj.batch_size
    return new_obj


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
            self.batch_value = {k: v * self.batch_size for k, v in m_value.items()}
        else:
            self.batch_value = m_value * self.batch_size
        return self

    def forward(self):
        if isinstance(self.batch_value, dict):
            return {k: v / self.batch_size for k, v in self.batch_value.items()}
        else:
            return self.batch_value / self.batch_size

    def add(self, obj):
        return add_patch_value(self, obj)

    def clone(self):
        """本方法已废弃"""
        new_obj = self.__class__(self.metric, self.batch_size, self.name)
        new_obj.batch_value = clone_value(self.batch_value)
        return new_obj


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
        self.metric = metric
        self.batch_size = batch_size

    def load(self, batch_preds, batch_targets):
        """
        Args:
            batch_pres:     一个mini_batch的模型预测
            batch_targets:  一个mini_batch的标签（当指标计算不需要标签时为空值）
        """
        self.batch_preds = [batch_preds] if isinstance(batch_preds, (list, tuple)) else [[batch_preds]]
        self.batch_targets = [batch_targets] if isinstance(batch_targets, (list, tuple)) else [[batch_targets]]
        return self

    def forward(self):
        try:
            # 合并各batch中的模型输出
            preds = [bpreds[0] if len(bpreds)==1 else torch.concat(bpreds, 0) for bpreds in zip(*self.batch_preds)]
            # 合并各batch的学习目标（标签）
            targets = None if self.batch_targets is None else \
                    [btargets[0] if len(btargets)==1 else torch.concat(btargets, 0) for btargets in zip(*self.batch_targets)]
        # 如果concat失败则放弃concat，以应对复杂的模型输出；而且要考虑到在GNN中batchsize为1的情况。
        except (RuntimeError, TypeError):
            preds = self.batch_preds
            targets = self.batch_targets

        preds = preds[0] if len(preds) == 1 else preds
        targets = targets[0] if len(targets) == 1 else targets
        return self.metric(preds, targets)

    def add(self, obj):
        assert self.metric is obj.metric, '相加的两个Patch的`metric`不一致'
        new_preds = self.batch_preds + obj.batch_preds
        if self.batch_targets != None:
            assert obj.batch_targets is not None, '相加的两个Patch的`batch_targets`其中一个为None！'
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
        if batch_preds.shape[1] == 1:
            num_classes = int((max(batch_targets) + 1).item())
        else:
            num_classes = batch_preds.shape[1]
        self.num_classes = num_classes
        self.confusion_matrix = confusion_matrix(batch_preds, batch_targets, num_classes)
        return self

    def forward(self):
        return self.metric(self.confusion_matrix)

    def add(self, obj):
        assert self.confusion_matrix.shape == obj.confusion_matrix.shape, '相加的两个Patch中数据的类别数量不相等！'

        # new_obj = self.__class__(self.metric, self.batch_size, self.name)
        new_obj = self                                      # 由于93行使用deepcopy复制了一个新的对象，因此没必要每次新建对象

        new_obj.num_classes = self.num_classes
        new_obj.confusion_matrix = self.confusion_matrix + obj.confusion_matrix
        return new_obj


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
        if batch_preds.shape[1] == 1:
            num_classes = int((max(batch_targets) + 1).item())
        else:
            num_classes = batch_preds.shape[1]
        self.num_classes = num_classes
        self.confusion_matrix = confusion_matrix(batch_preds, batch_targets, num_classes)
        return self

    def forward(self):
        return {self.metric2name[m]: getattr(self, m)() for m in self.metrics}

    def add(self, obj):
        assert self.confusion_matrix.shape == obj.confusion_matrix.shape, '相加的两个Patch中数据的类别数量不相等！'
        assert set(self.metrics) == set(obj.metrics), '相加的两个Patch的`metrics`不一致!'
        assert self.average == obj.average, '相加的两个Patch的`average`不一致!'

        # new_obj = self.__class__(self.metrics, self.average, self.beta, self.name)
        new_obj = self                                  # 由于93行使用deepcopy复制了一个新的对象，因此没必要每次新建对象

        new_obj.num_classes = self.num_classes
        new_obj.confusion_matrix = self.confusion_matrix + obj.confusion_matrix
        return new_obj

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
