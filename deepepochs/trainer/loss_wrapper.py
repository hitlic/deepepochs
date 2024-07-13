"""
@author: liuchen
"""

class LossWrapper:
    """
    1. 自动完成zero_grad、backward、opt.step等操作
    2. 配合实现梯度累积
    3. 实现回调
            on_before_backward    on_after_backward
            on_before_optimize    on_after_optimize
            on_train_metrics      on_val_metrics       on_test_metrics
    """
    def __init__(self, loss_fn, trainer):
        self.loss_fn = loss_fn
        self.trainer = trainer
        self.stage = None
        self.do_loss = None
        self.task = None

    def optimize(self):
        self.trainer.callbacks.trigger('before_optimize', trainer=self.trainer)
        self.trainer.opt.step()
        self.trainer.opt.zero_grad()
        self.trainer.callbacks.trigger('after_optimize', trainer=self.trainer)

    def __call__(self, model_out, batch_y, loss_adjust=1.0, grad_accumulate=False):
        """
        Args:
            model_out:          模型预测输出
            batch_y:            标签
            loss_adjust:        取值为sub_batch_size与batch_size的比例，以使累积梯度训练结果与正常训练一致（参见例22、23）
            grad_accumulate:    值为True时不调用optimize和on_metric_callback（参见例22、23）
        """
        if self.stage == 'train':
            # 计算损失
            loss = self.loss_fn(model_out, batch_y)

            # backward
            self.trainer.callbacks.trigger('before_backward', trainer=self.trainer, loss=loss)
            if self.trainer.accelerator is None:
                (loss * loss_adjust).backward()
            else:       # accelerate的backward
                self.trainer.accelerator.backward(loss * loss_adjust)
            self.trainer.callbacks.trigger('after_backward', trainer=self.trainer, loss=loss)

            # 优化、触发回调。累积梯度情况下需在累积结束后手动调用。
            if not grad_accumulate:
                self.optimize()                                     # 更新参数
                self.do_metric(loss, model_out, batch_y)   # 触发指标计算回调
        else:
            if self.do_loss:
                loss = self.loss_fn(model_out, batch_y)
            else:
                loss = None
            self.do_metric(loss, model_out, batch_y)       # 触发指标计算回调
        return loss

    def do_metric(self, loss, model_out, batch_y):
        """触发指标计算回调"""
        if loss is not None and hasattr(loss, 'detach'):
            loss = loss.detach().clone()
        if model_out is not None and hasattr(model_out, 'detach'):
            model_out = model_out.detach().clone()
        self.trainer.callbacks.trigger(f'{self.stage}_metrics', trainer=self.trainer, loss=loss, model_out=model_out, batch_y=batch_y, task=self.task)
