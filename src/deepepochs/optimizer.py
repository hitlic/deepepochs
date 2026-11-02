"""
@author: liuchen
"""
class Optimizer:
    def __init__(self, opt, scheduler=None, sched_on='epoch', sched_with_loss=False, monitor=None, on_stage='val'):
        """
        优化器组合，对优化器和学习率调度器进行统一管理。
        Args:
            opt:             torch.optim.*
            scheduler:       torch.optim.lr_scheduler.*
            sched_on:        学习率调整是每个epoch还是每个step
            sched_with_loss: scheduler.step方法是否需要损失作为参数（例如ReduceLROnPlateau）
            monitor:         epoch调度使用的完整指标键；None按唯一loss自动选择
            on_stage:        显式monitor的阶段，train或val；未验证轮次跳过val调度
        """
        self.opt = opt
        self.scheduler = scheduler
        assert sched_on in ['step', 'epoch'], '`sched_on`取值为"step"或"epoch"!'
        self.sched_on = sched_on
        self.sched_with_loss = sched_with_loss
        if on_stage not in ('train', 'val'):
            raise ValueError('on_stage必须是train或val。')
        self.monitor = monitor
        self.on_stage = on_stage

    def zero_grad(self):
        self.opt.zero_grad()

    def get_last_lr(self):
        return self.scheduler.get_last_lr() if self.scheduler is not None else None

    def step(self, at='step', loss=None, scaler=None):
        if at == 'step':
            if scaler is None or not scaler.is_enabled():
                self.opt.step()
            else:
                scaler.step(self.opt)
                # GradScaler 按优化器记录反缩放时的溢出；后续 value 裁剪
                # 即使将梯度改为有限值，也不改变本次跳步和调度判定。
                if any(value.item() for value in scaler._found_inf_per_device(self.opt).values()):
                    return
            if self.sched_on == 'step':
                self.sched_step(loss)
        elif at == 'epoch':
            if self.sched_on == 'epoch':
                self.sched_step(loss)
        else:
            raise ValueError('Optimizer.step方法的`at`参数取值为"step"或"epoch"')

    def step_epoch(self, train_metrics, val_metrics, validated):
        """epoch指标调度与各优化器自己的step损失窗口分别处理。"""
        if self.scheduler is None or self.sched_on != 'epoch':
            return
        loss = None
        if self.sched_with_loss:
            if self.monitor is not None:
                if self.on_stage == 'val' and not validated:
                    return
                metrics = val_metrics if self.on_stage == 'val' else train_metrics
                if self.monitor not in metrics:
                    raise ValueError(f'epoch调度指标不存在：{self.on_stage}/{self.monitor}')
                loss = metrics[self.monitor]
            else:
                loss = self._unique_loss(val_metrics)
                if loss is None:
                    loss = self._unique_loss(train_metrics)
        self.step(at='epoch', loss=loss)

    @staticmethod
    def _unique_loss(metrics):
        keys = [key for key in metrics if key == 'loss' or key.endswith('/loss')]
        if len(keys) > 1:
            raise ValueError('多个任务提供loss，请用Optimizer的monitor和on_stage明确调度指标。')
        return metrics[keys[0]] if keys else None

    def sched_step(self, loss):
        if self.scheduler is not None:
            if self.sched_with_loss:
                assert loss is not None, "学习率调度要求损失作为参数，但`train_step`和`evaluate_step`都没有返回`loss`！"
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

    def state_dict(self):
        sched_state = None if self.scheduler is None else self.scheduler.state_dict()
        return {'opt_state': self.opt.state_dict(), 'sched_state': sched_state}

    def load_state_dict(self, state):
        opt_state, sched_state = state['opt_state'], state['sched_state']
        self.opt.load_state_dict(opt_state)
        if sched_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(sched_state)

    @property
    def param_groups(self):
        return self.opt.param_groups

    def get_current_lr(self):
        for param_group in self.param_groups:
            return param_group['lr']


class Optimizers(list):
    """
    用于管理多个优化器组合（Optimizer），对多个优化器提供支持。
    """
    def zero_grad(self):
        for opt in self:
            opt.zero_grad()

    def get_last_lr(self):
        return [opt.get_last_lr() for opt in self]

    def step(self, at='step', loss=None):
        for opt in self:
            opt.step(at, loss)

    def step_epoch(self, train_metrics, val_metrics, validated):
        for opt in self:
            opt.step_epoch(train_metrics, val_metrics, validated)

    def state_dict(self):
        return [opt.state_dict() for opt in self]

    def load_state_dict(self, states):
        for opt, state in zip(self, states):
            opt.load_state_dict(state)

    def get_current_lr(self):
        return [opt.get_current_lr() for opt in self]
