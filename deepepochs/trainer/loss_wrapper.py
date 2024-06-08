"""
@author: liuchen
"""
from ..loops import concat, detach_clone


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

        self.total_loss = 0     # 用于实现累积梯度
        self.model_outs = []    # 用于实现累积梯度
        self.batch_ys = []      # 用于实现累积梯度

    def optimize(self):
        self.trainer.callbacks.trigger('before_optimize', trainer=self.trainer)
        self.trainer.opt.step()
        self.trainer.opt.zero_grad()
        self.trainer.callbacks.trigger('after_optimize', trainer=self.trainer)

    def __call__(self, model_out, batch_y, grad_accumulate=False):
        """
        Args:
            model_out:         模型预测输出
            batch_y:           标签
            grad_accumulate:   是否累积梯度
        """
        if self.stage == 'train':
            # 计算损失
            loss = self.loss_fn(model_out, batch_y)

            # backward
            self.trainer.callbacks.trigger('before_backward', trainer=self.trainer, loss=loss)
            if self.trainer.accelerator is None:
                (loss/self.trainer.grad_accumulate_steps).backward()
            else:       # accelerate的backward
                self.trainer.accelerator.backward(loss/self.trainer.grad_accumulate_steps)
            self.trainer.callbacks.trigger('after_backward', trainer=self.trainer, loss=loss)

            # 记录各sub-batch的总损失、模型输出、标签
            _loss = loss.detach().clone()
            self.total_loss += _loss * self.trainer.find_batch_size(model_out)
            self.model_outs.append(detach_clone(model_out))
            self.batch_ys.append(batch_y)

            # --- 累积梯度
            # 如果当前batch是中间sub_batch
            if grad_accumulate:
                if self.trainer.accelerator is not None: # DeepEpochs的梯度累积要求仅最后一个sub-batch优化
                    self.optimize()                      # Accelerate的梯度累积要求每个sub-batch都优化
                return _loss
            # 如果当前batch是最后一个sub_batch，或唯一的batch（没有使用梯度累积）
            else:
                self.optimize()
                # 计算平均损失，拼接多次累积度累积中的sub-batch的model_out和batch_y
                loss_4cbk = self.total_loss / sum(self.trainer.find_batch_size(o) for o in self.model_outs)

                try:
                    model_out_4cbk = self.model_outs[0] if len(self.model_outs) == 1 else concat(self.model_outs)
                    batch_y_4cbk = self.batch_ys[0] if len(self.batch_ys) == 1 else concat(self.batch_ys)
                # 如果concat失败则放弃concat，以应对复杂的模型输出；而且要考虑到在GNN中batchsize为1的情况。
                except RuntimeError:
                    model_out_4cbk = self.model_outs[0] if len(self.model_outs) == 1 else self.model_outs
                    batch_y_4cbk = self.batch_ys[0] if len(self.batch_ys) == 1 else self.batch_ys

                self.total_loss = 0
                self.model_outs = []
                self.batch_ys = []
        else:
            # 验证与测试不需要实现分批，如果需要的话可使用较小的batch_size
            model_out_4cbk = model_out
            batch_y_4cbk = batch_y
            if self.do_loss:
                loss_4cbk = self.loss_fn(model_out, batch_y)
            else:
                loss_4cbk = None
        self.trainer.callbacks.trigger(f'{self.stage}_metrics', trainer=self.trainer, loss=loss_4cbk, model_out=model_out_4cbk, batch_y=batch_y_4cbk, task=self.task)
        return loss_4cbk
