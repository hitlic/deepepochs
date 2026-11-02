"""损失、反向传播与参数更新：单卡和 DDP 使用同一条执行路径。"""
import torch
import torch.distributed as dist


class LossWrapper:
    def __init__(self, loss_fn, trainer):
        self.loss_fn = loss_fn
        self.trainer = trainer
        self.stage = None
        self.do_loss = None
        self.task = None
        self.last_loss = None
        self.optimizing_parameters = []
        self._normalized_parameters = set()
        self._unscaled_parameters = set()
        self._normalizer = None
        self._scaler_stepped = False
        self._stepped_optimizers = set()
        self._pending_loss = None
        self._last_loss_packet = None
        self._optimizer_losses = {}

    def begin_batch(self, new_group):
        """累积窗口内保持梯度尺度不变，窗口结束才统一归一化和更新。"""
        if new_group:
            self._normalized_parameters.clear()
            self._unscaled_parameters.clear()
            self._normalizer = None

    def finish_batch(self):
        # 多优化器共用同一次反向传播时，所有 optimizer.step 完成后才更新尺度。
        if self._scaler_stepped:
            self.trainer.scaler.update()
            self._scaler_stepped = False
            self._stepped_optimizers.clear()

    def discard_window(self):
        """丢弃中断窗口的梯度和统计，保留已经完成的参数更新。"""
        trainer = self.trainer
        trainer.opt.zero_grad()
        if self._scaler_stepped:
            # 已提交的优化器更新仍需完成本次尺度调整。
            self.finish_batch()
        elif self._unscaled_parameters:
            # 仅完成反缩放的窗口没有提交更新：保持尺度和增长计数，
            # 同时通过公开接口清除 GradScaler 的逐优化器阶段记录。
            trainer.scaler.update(new_scale=trainer.scaler.get_scale())

        self.last_loss = None
        self.optimizing_parameters = []
        self._pending_loss = None
        self._last_loss_packet = None
        self._optimizer_losses.clear()
        self._normalized_parameters.clear()
        self._unscaled_parameters.clear()
        self._normalizer = None
        self._scaler_stepped = False
        self._stepped_optimizers.clear()
        trainer._accumulation_count = 0
        trainer._accumulation_should_step = True

    def _route_loss(self, indices, loss):
        """optimize(index) 将最近一组 loss 调用归入对应优化器的窗口。

        中间累积批次也必须先归属再返回；多个优化器不会消费彼此的损失。
        同一次 backward 分别更新多个优化器时，可复用上一组损失统计。
        """
        if loss is not None:
            count = self.task.current_batch_size
            packet = (torch.as_tensor(loss, device=self.trainer.device).detach() * count, count)
        else:
            packet = self._pending_loss if self._pending_loss is not None else self._last_loss_packet
        if packet is not None:
            self._last_loss_packet = packet
            total, count = packet
            for index in indices:
                previous_total, previous_count = self._optimizer_losses.get(index, (0, 0))
                self._optimizer_losses[index] = (previous_total + total, previous_count + count)
        self._pending_loss = None

    def _scheduler_loss(self, index, optimizer):
        if optimizer.sched_on != 'step' or not optimizer.sched_with_loss:
            return self.last_loss
        packet = self._optimizer_losses.get(index)
        if packet is None:
            return self.last_loss
        total, count = packet
        totals = torch.stack((torch.as_tensor(total, device=self.trainer.device).float(),
                              torch.tensor(float(count), device=self.trainer.device)))
        if self.trainer.distributed.is_distributed:
            dist.all_reduce(totals)
        return totals[0] / totals[1]

    def optimize(self, optimize_index=None, loss=None):
        """更新全部或指定优化器；累积窗口的中间批次只保留梯度。"""
        trainer = self.trainer
        indices = list(range(len(trainer.opt))) if optimize_index is None else [optimize_index]
        self._route_loss(indices, loss)
        if not trainer._accumulation_should_step:
            return
        optimizers = [trainer.opt[index] for index in indices]
        parameters = []
        seen = set()
        for opt in optimizers:
            for group in opt.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))

        if trainer.gradient_accumulation_steps > 1:
            if self._normalizer is None:
                count = torch.tensor(float(trainer._accumulation_count), device=trainer.device)
                if trainer.distributed.is_distributed:
                    dist.all_reduce(count)
                # DDP 已对进程梯度取平均。先累积各批损失之和，再除以
                # 全局样本数/world_size，尾窗口和不等长尾批也使用真实计数。
                self._normalizer = count / trainer.world_size
            for parameter in parameters:
                if id(parameter) not in self._normalized_parameters:
                    parameter.grad.div_(self._normalizer)
                    self._normalized_parameters.add(id(parameter))

        if trainer.scaler.is_enabled() and any(id(opt.opt) in self._stepped_optimizers for opt in optimizers):
            # 同一 step 允许对一个优化器多次更新。新的 backward 已使用旧尺度，
            # 先结束上一次更新，再把仍待更新的梯度换算到新尺度。
            old_scale = trainer.scaler.get_scale()
            self.finish_batch()
            ratio = trainer.scaler.get_scale() / old_scale
            for parameter in trainer.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(ratio)
            self._unscaled_parameters.clear()

        if trainer.scaler.is_enabled():
            scale = trainer.scaler.get_scale()
            for opt in optimizers:
                opt_parameters = [p for group in opt.param_groups for p in group['params'] if p.grad is not None]
                # 现有接口允许优化器共享参数。GradScaler 按优化器反缩放，
                # 已反缩放的共享梯度先恢复尺度，保证每个梯度最终只除一次。
                for parameter in opt_parameters:
                    if id(parameter) in self._unscaled_parameters:
                        parameter.grad.mul_(scale)
                if opt_parameters:
                    trainer.scaler.unscale_(opt.opt)
                    self._unscaled_parameters.update(id(p) for p in opt_parameters)

        self.optimizing_parameters = parameters
        trainer.callbacks.trigger('before_optimize', trainer=trainer, optimize_index=optimize_index)
        # 每个优化器独立归约其窗口，所有进程得到一致的调度损失。
        for index, opt in zip(indices, optimizers):
            sched_loss = self._scheduler_loss(index, opt)
            has_gradient = any(p.grad is not None for group in opt.param_groups for p in group['params'])
            if trainer.scaler.is_enabled() and not has_gradient:
                self._optimizer_losses.pop(index, None)
                continue
            # 调度器可能在参数更新后抛出异常；先登记AMP调用，退出时仍完成尺度收尾。
            self._scaler_stepped = self._scaler_stepped or trainer.scaler.is_enabled()
            opt.step(loss=sched_loss, scaler=trainer.scaler)
            if trainer.scaler.is_enabled():
                self._stepped_optimizers.add(id(opt.opt))
            self._optimizer_losses.pop(index, None)
        for opt in optimizers:
            opt.zero_grad()
        # zero_grad 后下一次 backward 是新梯度，不能沿用上次的归一化标记。
        self._normalized_parameters.difference_update(seen)
        self._unscaled_parameters.difference_update(seen)
        trainer.callbacks.trigger('after_optimize', trainer=trainer, optimize_index=optimize_index)

    def __call__(self, model_out, batch_y, loss_adjust=1.0, do_optimize=True, do_metric=True):
        """计算损失；训练时反向传播，默认请求参数更新并记录指标。

        do_optimize=False保留反向传播，只将更新时机交给optimize()；
        do_metric=False跳过本次指标处理。参数实际更新后清空相应优化器的梯度。
        验证与测试仅计算损失和指标，不执行反向或参数更新。

        loss_adjust为手动子批的权重。自动跨批累积在此处乘batch样本数，
        更新时再按窗口的真实样本数归一化。
        """
        trainer = self.trainer
        loss = None
        if self.stage == 'train' or self.do_loss:
            with trainer.autocast():
                loss = self.loss_fn(model_out, batch_y)
        if self.stage == 'train':
            assert isinstance(loss, torch.Tensor) and loss.ndim == 0, '损失函数必须返回标量 Tensor！'
            self.last_loss = loss.detach()
            count = loss_adjust * self.task.current_batch_size
            weighted_loss = self.last_loss * count
            previous_total, previous_count = self._pending_loss if self._pending_loss is not None else (0, 0)
            self._pending_loss = (previous_total + weighted_loss, previous_count + count)
            trainer.callbacks.trigger('before_backward', trainer=trainer, loss=loss.detach().clone())
            weight = self.task.current_batch_size if trainer.gradient_accumulation_steps > 1 else 1
            trainer.scaler.scale(loss * loss_adjust * weight).backward()
            trainer.callbacks.trigger('after_backward', trainer=trainer, loss=loss.detach().clone())
            if do_optimize:
                self.optimize()
        if do_metric:
            self.do_metric(loss, model_out, batch_y)
        return loss

    def do_metric(self, loss=None, model_out=None, batch_y=None):
        if loss is not None and hasattr(loss, 'detach'):
            loss = loss.detach().clone()
        if model_out is not None and hasattr(model_out, 'detach'):
            model_out = model_out.detach().clone()
        self.trainer.callbacks.trigger(f'{self.stage}_metrics', trainer=self.trainer,
                                       loss=loss, model_out=model_out, batch_y=batch_y, task=self.task)
