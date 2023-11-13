from functools import partial
from .callback import Callback
from ..loops import listify, check_path
from torch.utils.tensorboard import SummaryWriter
from os import path as osp
from .log import run_tensorboard


class ListContainer():
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):  # idx为bool列表，返回idx中值为true的位置对应的元素
            assert len(idx) == len(self)  # bool mask
            return [obj for m, obj in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, obj):
        self.items[i] = obj

    def __delitem__(self, i):
        del(self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10:
            res = res[:-1] + '...]'
        return res


class Hook():
    def __init__(self, module, fun, mode, idx):
        if mode == 'forward':
            self.hook = module.register_forward_hook(partial(fun, self))
        else:
            self.hook = module.register_full_backward_hook(partial(fun, self))
        m_str = str(module).replace('\n', '').replace(' ', '')
        self.name = f"{idx}-{m_str}"

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, model, fun, mode):
        """
        Args:
            model: pytorch module
            fun: hook function
            mode: forward hook or backward hook
        """
        assert mode in ['forward', 'backward']
        if mode == 'forward':
            modules = [(i, m) for i, m in enumerate(model.modules()) if len(list(m.children())) == 0]
        else:
            modules = [(i, m) for i, m in enumerate(model.modules()) if len(list(m.children())) == 0 and self.param_num(m) > 0]
        super().__init__([Hook(m, fun, mode, i) for i, m in modules])

    def param_num(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


class AnalyzeCallback(Callback):
    def __init__(self, mode='backward', log_dir='./logs'):
        """
        Args:
            mode: 'forward' 分析各层前向输出 or 'backward' 分析各层反向梯度
        """
        super().__init__()
        assert mode in ['forward', 'backward'], '`mode` must be "forward" or "backward"!'
        self.mode = mode
        self.log_dir = log_dir
        self.global_step = 0

    def on_before_fit(self, trainer, epochs):
        log_dir = osp.join(self.log_dir, trainer.running_id)
        check_path(log_dir)
        logger = getattr(trainer, 'logger', None)
        if logger is None:
            self.logger = SummaryWriter(log_dir=log_dir)
            trainer.logger = self.logger
        else:
            self.logger = logger

        def output_stats(hook, module, inputs, outputs):
            if isinstance(outputs, tuple):  # backward hook
                outputs = outputs[0]
            hook.mean = outputs[0].data.mean()
            hook.std = outputs[0].data.std()
            hook.data = outputs[0].data
        self.hooks = Hooks(trainer.model, output_stats, self.mode)

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.global_step += 1
        mode = 'FORWARD' if self.mode == 'forward' else 'BACKWARD'
        mean_dict = {h.name: h.mean for h in self.hooks}
        std_dict = {h.name: h.std for h in self.hooks}
        for k, v in mean_dict.items():
            self.logger.add_scalar(f'{mode}-mean-{k}', v, global_step=self.global_step)
        for k, v in std_dict.items():
            self.logger.add_scalar(f'{mode}-std-{k}', v, global_step=self.global_step)
        for h in self.hooks:
            self.logger.add_histogram(f'{mode}-hist-{h.name}', h.data, global_step=self.global_step)

    def on_after_fit(self, trainer):
        self.hooks.remove()
        self.logger.close()

    def run_tensorboard(self):
        run_tensorboard(self.log_dir)
