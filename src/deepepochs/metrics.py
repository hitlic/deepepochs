"""
@author: liuchen
"""
import torch

from .loops import ddict


def confusion_matrix(preds, targets, num_classes):
    """
    Args:
        preds:      预测向量，可为binary或多维概率分布
        targets:    标签向量，可为one-hot或非one-hot的
        num_class:  类别数量
    """
    if (preds.dim()==1 or preds.shape[-1]==1) and num_classes==2:  # 当预测为binary时
        preds = preds.unsqueeze(-1) if preds.dim()==1 else preds
        preds = torch.concat([1-preds, preds], dim=-1)
    preds = preds.argmax(dim=-1).flatten().int()

    if targets.dim() > 1 and targets.shape[-1] > 1: # 当targets为one-hot时
        targets = targets.argmax(dim=1).int()
    else:
        targets = targets.flatten().int()
    cm = torch.zeros([num_classes, num_classes], dtype=preds.dtype, device=preds.device)
    one = torch.tensor([1], dtype=preds.dtype, device=preds.device)
    return cm.index_put_((targets, preds), one, accumulate=True)


def _class_counts(c_mat):
    return c_mat.diagonal(), c_mat.sum(dim=1), c_mat.sum(dim=0)


def cmats_and_weights(c_mat):
    """获取各类别的混淆矩阵和权值"""
    tp, support, predicted = _class_counts(c_mat)
    fn = support - tp
    fp = predicted - tp
    tn = support.sum() - tp - fn - fp
    cmats = [ddict(TP=t, FN=n, FP=p, TN=u) for t, n, p, u in zip(tp, fn, fp, tn)]
    return cmats, list((support / support.sum()).unbind())


def _divide(numerator, denominator):
    return torch.where(denominator == 0, 0, numerator / denominator)


def _metric_counts(preds, targets, conf_mat, average):
    __check_params(preds, targets, average, conf_mat)
    if conf_mat is None:
        conf_mat = confusion_matrix(preds, targets, get_class_num(preds, targets))
    tp, support, predicted = _class_counts(conf_mat)
    if average == 'micro':
        return tp.sum(), support.sum(), predicted.sum()
    return tp, support, predicted


def _average(values, support, average):
    if average == 'micro':
        return values
    if average == 'macro':
        return values.mean()
    return (values * (support / support.sum())).sum()


def precision_fn(c_mat):
    if c_mat.TP + c_mat.FP == 0:
        return 0
    return c_mat.TP/(c_mat.TP + c_mat.FP)


def recall_fn(c_mat):
    if c_mat.TP + c_mat.FN == 0:
        return 0
    return c_mat.TP/(c_mat.TP + c_mat.FN)


def fbeta_fn(c_mat, beta):
    p = precision_fn(c_mat)
    r = recall_fn(c_mat)
    if p + r == 0:
        return 0
    return (1 + beta**2) * (p*r)/(beta**2 * p + r)


def __check_params(preds, targets, average, conf_mat):
    """检查参数"""
    assert average in ['micro', 'macro', 'weighted'], "`average`取值为'micro', 'macro'或'weighted'"
    if conf_mat is None:
        assert preds is not None and targets is not None, "请提供`conf_mat`或(`preds`,`targets`)"
    else:
        assert preds is None and targets is None, "`conf_mat`不为空！请检查参数!"


def get_class_num(preds, targets):
    """获取类别数量"""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    return 2 if preds.dim() == 1 or preds.shape[-1] == 1 else preds.shape[-1]


def accuracy(preds=None, targets=None, conf_mat=None):
    """
    基于conf_mat或(preds, targets)计算Accuracy。conf_mat和(preds, targets)不能同时为None。
    Args:
        conf_mat: 混淆矩阵
        preds:    预测值
        targets:  标签
    """
    __check_params(preds, targets, average='micro', conf_mat=conf_mat)
    if conf_mat is None:
        num_class = get_class_num(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)

    return conf_mat.diagonal().sum() / conf_mat.sum()


def recall(preds=None, targets=None, conf_mat=None, average='micro'):
    """
    基于conf_mat或(preds, targets)计算Recall。conf_mat和(preds, targets)不能同时为None。
    Args:
        conf_mat: 混淆矩阵
        preds:    预测值
        targets:  标签
        average:  平均方式
    """
    tp, support, _ = _metric_counts(preds, targets, conf_mat, average)
    return _average(_divide(tp, support), support, average)


def precision(preds=None, targets=None, conf_mat=None, average='micro'):
    """
    基于conf_mat或(preds, targets)计算Precision。conf_mat和(preds, targets)不能同时为None。
    Args:
        conf_mat: 混淆矩阵
        preds:    预测值
        targets:  标签
        average:  平均方式
    """
    tp, support, predicted = _metric_counts(preds, targets, conf_mat, average)
    return _average(_divide(tp, predicted), support, average)


def fbeta(preds=None, targets=None, conf_mat=None, beta=1, average='micro'):
    """
    基于conf_mat或(preds, targets)计算F-beta。conf_mat和(preds, targets)不能同时为None。
    Args:
        conf_mat: 混淆矩阵
        preds:    预测值
        targets:  标签
        average:  平均方式
    """
    tp, support, predicted = _metric_counts(preds, targets, conf_mat, average)
    p = _divide(tp, predicted)
    r = _divide(tp, support)
    values = torch.where(p + r == 0, 0, (1 + beta**2) * (p * r) / (beta**2 * p + r))
    return _average(values, support, average)


def f1(preds=None, targets=None, conf_mat=None, average='micro'):
    """
    基于conf_mat或(preds, targets)计算F1。conf_mat和(preds, targets)不能同时为None。
    Args:
        conf_mat: 混淆矩阵
        preds:    预测值
        targets:  标签
        average:  平均方式
    """
    return fbeta(preds, targets, conf_mat, 1, average=average)
