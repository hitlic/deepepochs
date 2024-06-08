from .callback import Callback
from torch.nn.utils.clip_grad import clip_grad_value_, clip_grad_norm_
from functools import partial


class GradClipCallback(Callback):
    def __init__(self, max_value, norm_type=None):
        """梯度截断。
        Args:
            max_value (float):       最大取值
            norm_type (None, float): 取值为None或正数p-norm，None表示直接对取值进行剪裁，否则通过向量的指定类型范数进行剪裁
        """
        assert norm_type is None or norm_type > 0, 'The value of norm_type is None or a positive integer.'
        self.max_value = max_value
        if norm_type is None:
            self.clip = clip_grad_value_
        else:
            self.clip = partial(clip_grad_norm_, norm_type=norm_type)
        super().__init__()

    def on_after_optimize(self, trainer):
        self.clip(trainer.model.parameters(), self.max_value)
        return super().on_after_optimize(trainer)
