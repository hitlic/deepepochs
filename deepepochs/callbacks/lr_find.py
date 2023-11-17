from .callback import Callback, CallbackException
from ..loops import StopLoopException
from ..optimizer import Optimizers
from matplotlib import pyplot as plt


class LRFindCallback(Callback):
    def __init__(self, max_batch=100, min_lr=1e-6, max_lr=5, opt_id=None):
        """
        Args:
            max_batch: 执行的mini-batch数量
            min_lr: 最小学习率
            max_lr: 最大学习率
            opt_id: 优化器序号，当使用多个优化器时需指定
        """
        super().__init__()
        self.max_batch, self.min_lr, self.max_lr = max_batch, min_lr, max_lr
        self.best_loss = 1e9
        self.lrs = []
        self.losses = []
        self.opt_id = opt_id

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        if isinstance(trainer.opt, Optimizers):
            if self.opt_id is None:
                raise CallbackException("训练中使用了多个优化器，参数opt_id未指定！")
            opt = trainer.opt[self.opt_id]
        else:
            opt = trainer.opt
        pos = (batch_idx + 1)/self.max_batch
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in opt.param_groups:
            pg['lr'] = lr
        self.lrs.append(lr)

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        batch_loss = metrics['loss']
        if batch_loss < self.best_loss:
            self.best_loss = batch_loss
        self.losses.append(batch_loss.item())
        if batch_idx+1 >= self.max_batch or batch_loss > self.best_loss*10:
            raise StopLoopException()

    def on_after_fit(self, trainer):
        plt.plot(self.lrs, self.losses)
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.show()
