
class Optimizer:
    def __init__(self, opt, scheduler=None, sched_on='epoch', sched_with_loss=False):
        """
        优化器组合，对优化器和学习率调度器进行统一管理。
        Args:
            opt:             torch.optim.*
            scheduler:       torch.optim.lr_scheduler.*
            sched_on:        学习率调整是每个epoch还是每个step
            sched_with_loss: scheduler.step方法是否需要损失作为参数（例如ReduceLROnPlateau）
        """
        self.opt = opt
        self.scheduler = scheduler
        assert sched_on in ['step', 'epoch'], '`sched_on`取值为"step"或"epoch"!'
        self.sched_on = sched_on
        self.sched_with_loss = sched_with_loss

    def zero_grad(self):
        self.opt.zero_grad()

    def get_last_lr(self):
        return self.scheduler.get_last_lr() if self.scheduler is not None else None

    def step(self, at='step', loss=None):
        if at == 'step':
            self.opt.step()
            if self.sched_on == 'step':
                self.sched_step(loss)
        elif at == 'epoch':
            if self.sched_on == 'epoch':
                self.sched_step(loss)
        else:
            raise ValueError('Optimizer.step方法的`at`参数取值为"step"或"epoch"')

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
            self.scheduler.load_state_dict(opt_state)

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

    def state_dict(self):
        return [opt.state_dict() for opt in self]

    def load_state_dict(self, states):
        for opt, state in zip(self, states):
            opt.load_state_dict(state)

    def get_current_lr(self):
        return [opt.get_current_lr() for opt in self]
