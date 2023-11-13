from .loops import confusion_matrix, get_cmats, recall_fn, precision_fn, fbeta_fn, sum_dicts


def __check_params(preds, targets, average, conf_mat):
    """检查参数"""
    assert average in ['micro', 'macro', 'weighted'], "`average`取值为'micro', 'macro'或'weighted'"
    if conf_mat is None:
        assert preds is not None and targets is not None, "请提供`conf_mat`或(`preds`,`targets`)"


def __num_class(preds, targets):
    """获取类别数量"""
    if preds.shape[1] == 1:
        num_classes = int((max(targets) + 1).item())
    else:
        num_classes = preds.shape[1]
    return num_classes


def __weights(c_mats):
    """获取各类别权值"""
    weights = [mat.TP+mat.FN for mat in c_mats]
    w_sum = sum(weights)
    weights = [w/w_sum for w in weights]
    return weights


def accuracy(preds=None, targets=None, conf_mat=None):
    __check_params(preds, targets, average='micro', conf_mat=conf_mat)
    if conf_mat is None:
        num_class = __num_class(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    return sum(conf_mat[i, i] for i in range(num_class))/conf_mat.sum()


def recall(preds=None, targets=None, average='micro', conf_mat=None):
    __check_params(preds, targets, average=average, conf_mat=conf_mat)
    if conf_mat is None:
        num_class = __num_class(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    c_mats = get_cmats(conf_mat)
    weights = __weights(c_mats)

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
        num_class = __num_class(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    c_mats = get_cmats(conf_mat)
    weights = __weights(c_mats)

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
        num_class = __num_class(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_class)
    else:
        num_class = conf_mat.shape[0]

    c_mats = get_cmats(conf_mat)
    weights = __weights(c_mats)

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
