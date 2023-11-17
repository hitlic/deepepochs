from .loops import sum_dicts, ddict
import torch
from functools import lru_cache


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def cmats_and_weights(c_mat):
    """获取各类别的混淆矩阵和权值"""
    if c_mat.shape[0] == 2:
        c_mat = ddict({
            'TP': c_mat[0][0],
            'FN': c_mat[0][1],
            'FP': c_mat[1][0],
            'TN': c_mat[1][1]
        })
        cmats = [c_mat]
    else:
        cmats = [ddict(__get_cmat_i(i, c_mat)) for i in range(c_mat.shape[0])]
    return cmats, __weights(cmats)


def __weights(c_mats):
    """获取各类别权值（TP+FN）"""
    weights = [mat.TP+mat.FN for mat in c_mats]
    w_sum = sum(weights)
    weights = [w/w_sum for w in weights]
    return weights


def __get_cmat_i(c_id, c_mat):
    TP = c_mat[c_id, c_id]
    FN = c_mat[c_id].sum() - TP
    FP = c_mat[:, c_id].sum() - TP
    TN = c_mat.sum() - TP - FN - FP
    return ddict({'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN})


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


def get_class_num(preds, targets):
    """获取类别数量"""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if preds.shape[1] == 1:
        num_classes = int((max(targets) + 1).item())
    else:
        num_classes = preds.shape[1]
    return num_classes


def accuracy(preds=None, targets=None, conf_mat=None):
    __check_params(preds, targets, average='micro', conf_mat=conf_mat)
    if conf_mat is None:
        num_class = get_class_num(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    return sum(conf_mat[i, i] for i in range(num_class))/conf_mat.sum()


def recall(preds=None, targets=None, average='micro', conf_mat=None):
    __check_params(preds, targets, average=average, conf_mat=conf_mat)
    if conf_mat is None:
        num_class = get_class_num(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    c_mats, weights = cmats_and_weights(conf_mat)

    if average == 'micro':
        return recall_fn(sum_dicts(c_mats))
    elif average == 'macro':
        return sum(recall_fn(mat) for mat in c_mats)/num_class
    else:
        ps = [recall_fn(mat) for mat in c_mats]
        v = sum(p*w for p, w in zip(ps, weights))
        return v


def precision(preds=None, targets=None, average='micro', conf_mat=None):
    __check_params(preds, targets, average=average, conf_mat=conf_mat)
    if conf_mat is None:
        num_class = get_class_num(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    c_mats, weights = cmats_and_weights(conf_mat)

    if average == 'micro':
        return precision_fn(sum_dicts(c_mats))
    elif average == 'macro':
        return sum(precision_fn(mat) for mat in c_mats)/num_class
    else:
        ps = [precision_fn(mat) for mat in c_mats]
        return sum(p*w for p, w in zip(ps, weights))


def fbeta(preds=None, targets=None, beta=1, average='micro', conf_mat=None):
    __check_params(preds, targets, average=average, conf_mat=conf_mat)
    if conf_mat is None:
        num_class = get_class_num(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    c_mats, weights = cmats_and_weights(conf_mat)

    if average == 'micro':
        return fbeta_fn(sum_dicts(c_mats), beta)
    elif average == 'macro':
        return sum(fbeta_fn(mat, beta) for mat in c_mats)/num_class
    else:
        ps = [fbeta_fn(mat, beta) for mat in c_mats]
        v = sum(p*w for p, w in zip(ps, weights))
        return v


def f1(preds=None, targets=None, average='micro', conf_mat=None):
    return fbeta(preds, targets, 1, average=average, conf_mat=conf_mat)
