"""
@author: hitlic
"""
from ..loops import ValuePatch, TensorPatch, MeanPatch, ConfusionPatch, Checker  # pylint: disable=W0611
from .learner_base import LearnerBase


class Learner(LearnerBase):
    def train_step(self, batch_x, batch_y):
        """
        TODO: 非常规训练可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        self.opt.zero_grad()
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)
        self.callbacks.trigger('before_backward', learner=self)
        loss.backward()
        self.callbacks.trigger('after_backward', learner=self)
        self.opt.step()

        results = {'loss': ValuePatch(loss.detach(), len(model_out))}
        for m in self.metrics:
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results

    def evaluate_step(self, batch_x, batch_y):
        """
        TODO: 非常规验证或测试可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)

        results = {'loss': ValuePatch(loss.detach(), len(model_out))}
        for m in self.metrics:
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results
