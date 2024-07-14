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

    def optimize(self, optimize_index=None):
        """
        Args:
            optimize_index: 优化器索引，当使用多个优化器时仅调用由optimize_index指定的优化器；None表示调用全部优化器.
        """
        self.trainer.callbacks.trigger('before_optimize', trainer=self.trainer, optimize_index=optimize_index)
        if optimize_index is None:
            self.trainer.opt.step()
            self.trainer.opt.zero_grad()
        else:
            self.trainer.opt[optimize_index].step()
            self.trainer.opt[optimize_index].zero_grad()
        self.trainer.callbacks.trigger('after_optimize', trainer=self.trainer, optimize_index=optimize_index)

    def __call__(self, model_out, batch_y, loss_adjust=1.0, do_optimize=True, do_metric=True):
        """
        Args:
            model_out:      模型预测输出
            batch_y:        标签
            loss_adjust:    取值为sub_batch_size与batch_size的比例，以使累积梯度训练结果与正常训练一致（参见例22、23）
            do_optimize:    是否调用优化器优化模型
            do_metric:      是否触发处理metric的回调
        """
        if self.stage == 'train':
            # 计算损失
            loss = self.loss_fn(model_out, batch_y)
            # backward
            self.trainer.callbacks.trigger('before_backward', trainer=self.trainer, loss=loss.detach())
            if self.trainer.accelerator is None:
                (loss * loss_adjust).backward()
            else:       # accelerate的backward
                self.trainer.accelerator.backward(loss * loss_adjust)
            self.trainer.callbacks.trigger('after_backward', trainer=self.trainer, loss=loss.detach())

            if do_optimize:
                self.optimize()                                     # 更新参数
        else:
            if self.do_loss:
                loss = self.loss_fn(model_out, batch_y)
            else:
                loss = None

        if do_metric:
            self.do_metric(loss, model_out, batch_y)   # 触发指标计算回调
        return loss

    def do_metric(self, loss=None, model_out=None, batch_y=None):
        """触发指标计算回调"""
        if loss is not None and hasattr(loss, 'detach'):
            loss = loss.detach()
        if model_out is not None and hasattr(model_out, 'detach'):
            model_out = model_out.detach()
        self.trainer.callbacks.trigger(f'{self.stage}_metrics', trainer=self.trainer, loss=loss, model_out=model_out, batch_y=batch_y, task=self.task)
