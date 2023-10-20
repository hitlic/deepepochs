from .loops import TrainerBase, ValuePatch, TensorPatch, Checker  # pylint: disable=W0611


class Trainer(TrainerBase):
    def train_step(self, batch_x, batch_y):
        """
        TODO: 非常规训练可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为封装了指标和数据的ValuePatch或者Patch。
        """
        self.opt.zero_grad()
        model_out = self.model(batch_x)
        loss = self.loss(model_out, batch_y)
        loss.backward()
        self.opt.step()

        results = {'loss': ValuePatch(loss.detach(), len(model_out))}
        for m in self.metrics:
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results

    def evaluate_step(self, batch_x, batch_y):
        """
        TODO: 非常规验证或测试可修改本方法中的代码。
        注意：本方法返回一个字典，键为指标名，值为Patch对象（封装了指标和数据）。
        """
        model_out = self.model(batch_x)
        loss = self.loss(model_out, batch_y)

        results = {'loss': ValuePatch(loss.detach(), len(model_out))}
        for m in self.metrics:
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results
