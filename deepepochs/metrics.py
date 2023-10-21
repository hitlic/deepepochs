from typing import Literal
from .loops import PatchBase, sum_dicts, ddict
import torch


class ConfusionPatch(PatchBase):
    def __init__(self, batch_preds, batch_targets,
                 metrics=('accuracy', 'precision', 'recall', 'f1', 'fbeta'),
                 average: Literal['micro', 'macro', 'weighted']='micro', beta=1.0):
        """
        能够累积计算基于混淆矩阵的指标，包括'accuracy', 'precision', 'recall', 'f1', 'fbeta'等。
        Args:
            batch_preds:    模型预测
            batch_targets:  标签
            metrics:        需计算的标签，'accuracy', 'precision', 'recall', 'f1', 'fbeta'中的一个或多个
            average:        多分类下的平均方式'micro', 'macro', 'weighted'之一
            beta:           F_beta中的beta
        """
        super().__init__()
        if isinstance(metrics, str):
            metrics = [metrics]

        assert set(metrics) <= set(['accuracy', 'precision', 'recall', 'f1', 'fbeta']),\
                "未知`metrics`！可取值为{'accuracy', 'precision', 'recall', 'f1', 'fbeta'}的子集！"
        assert average in ['micro', 'macro', 'weighted'], "`average`取值为['micro', 'macro', 'weighted']之一！"
        self.name = {'accuracy': '.acc', 'recall': '.r', 'precision': '.p', 'f1': '.f1', 'fbeta': 'fb'}

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
        return {self.name[m]: getattr(self, m)(c_mats, weights) for m in self.metrics}

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
