from os import path as osp

import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from ..loops import check_path, listify
from .callback import Callback
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
    def __init__(self, module, fun, mode, idx, sample=None):
        self.updated = False
        self.tensor_hooks = []

        def record_output(module, inputs, outputs):
            if sample is not None and not sample():
                return
            if mode == 'forward':
                fun(self, module, inputs, outputs)
            else:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.requires_grad:
                    # Tensor hooks 不引入输出视图，后续层仍可执行原位运算。
                    handle = outputs.register_hook(lambda grad: fun(self, module, inputs, grad))
                    self.tensor_hooks.append(handle)

        self.hook = module.register_forward_hook(record_output)
        m_str = str(module).replace('\n', '').replace(' ', '')
        self.name = f"{idx}-{m_str}"

    def clear_batch(self):
        for handle in self.tensor_hooks:
            handle.remove()
        self.tensor_hooks.clear()
        self.updated = False

    def remove(self):
        self.hook.remove()
        self.clear_batch()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, model, fun, mode, layers=None, sample=None):
        """
        Args:
            model: pytorch module
            fun: hook function
            mode: forward hook or backward hook
            layers: 精确模块名序列；None 沿用各模式的叶子层选择规则。
        """
        super().__init__([])
        assert mode in ['forward', 'backward']
        named_modules = list(model.named_modules())
        if layers is None:
            modules = [(i, m) for i, (_, m) in enumerate(named_modules)
                       if len(list(m.children())) == 0
                       and (mode == 'forward' or self.param_num(m) > 0)]
        else:
            names = dict(named_modules)
            unknown = [name for name in layers if name not in names]
            if unknown:
                raise ValueError(f'未知层名: {unknown!r}')
            modules = [(name or '<root>', m) for name, m in named_modules if name in layers]
        self.items = [Hook(m, fun, mode, name, sample) for name, m in modules]

    def param_num(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove() # type: ignore
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


class AnalyzeCallback(Callback):
    def __init__(self, mode='backward', log_dir='./logs', every_n_batches=1, layers=None):
        """
        Args:
            mode: 'forward' 分析各层前向输出 or 'backward' 分析各层反向梯度
            every_n_batches: 从首个训练 batch 开始，每隔指定数量的 batch 采样。
            layers: 原始模型 named_modules() 的精确名称序列，空字符串代表根模块。
                None 沿用叶子层规则；明确名称可选择父模块，标签使用该模块路径。
        """
        # 在 LogCallback 保存模型图后重置采样，避免将图追踪的输出计入训练 batch。
        super().__init__(priority=2)
        assert mode in ['forward', 'backward'], '`mode` must be "forward" or "backward"!'
        if isinstance(every_n_batches, bool) or not isinstance(every_n_batches, int) or every_n_batches < 1:
            raise ValueError('every_n_batches必须为正整数。')
        if isinstance(layers, str):
            raise TypeError('layers必须为模块名称序列或None。')
        self.every_n_batches = every_n_batches
        self.layers = None if layers is None else list(layers)
        self.mode = mode
        self.log_dir = log_dir
        self.global_step = 0

    def on_before_fit(self, trainer, epochs):
        if not trainer.main_process:
            return
        def output_stats(hook, module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # BF16 转 FP32 兼容 TensorBoard；独立存储保留采样时的分布。
            data = outputs.detach()
            data = data.float() if data.dtype == torch.bfloat16 else data.clone()
            hook.mean = data.mean()
            hook.std = data.std()
            hook.data = data
            hook.updated = True
        model = trainer.model
        if self.layers is not None and hasattr(trainer, 'distributed'):
            model = trainer.distributed.unwrap_serialization_model(model.model)
        self.hooks = Hooks(model, output_stats, self.mode, self.layers, self._should_sample)
        log_dir = osp.join(self.log_dir, trainer.running_id)
        check_path(log_dir)
        logger = getattr(trainer, 'logger', None)
        if logger is None:
            self.logger = SummaryWriter(log_dir=log_dir)
            trainer.logger = self.logger
        else:
            self.logger = logger

    def _should_sample(self):
        return self.global_step % self.every_n_batches == 0

    def state_dict(self):
        return {'global_step': self.global_step}

    def load_state_dict(self, state_dict):
        self.global_step = state_dict['global_step']

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        if not trainer.main_process:
            return
        for hook in self.hooks:
            hook.clear_batch()

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        if not trainer.main_process:
            return
        self.global_step += 1
        mode = 'FORWARD' if self.mode == 'forward' else 'BACKWARD'
        active_hooks = [h for h in self.hooks if h.updated]
        mean_dict = {h.name: h.mean for h in active_hooks}
        std_dict = {h.name: h.std for h in active_hooks}
        for k, v in mean_dict.items():
            self.logger.add_scalar(f'{mode}-mean-{k}', v, global_step=self.global_step)
        for k, v in std_dict.items():
            self.logger.add_scalar(f'{mode}-std-{k}', v, global_step=self.global_step)
        for h in active_hooks:
            self.logger.add_histogram(f'{mode}-hist-{h.name}', h.data, global_step=self.global_step)
        for h in self.hooks:
            h.clear_batch()

    def remove_hooks(self):
        """移除分析 hooks；尚未开始训练或已移除时也可以调用。"""
        hooks = getattr(self, 'hooks', None)
        if hooks is not None:
            hooks.remove()

    def on_after_fit(self, trainer):
        self.remove_hooks()

    def run_tensorboard(self):
        run_tensorboard(self.log_dir)
