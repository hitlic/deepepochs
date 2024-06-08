"""
@author: liuchen
"""

class ModelWrapper:
    """
    用于实现回调：
        on_before_train_forward    on_after_train_forward
        on_before_val_forward      on_after_val_forward
        on_before_test_forward     on_after_test_forward
    """
    def __init__(self, model, trainer):
        # self.model = torch.compile(model)
        self.model = model
        self.trainer = trainer
        self.stage = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwds):
        self.trainer.callbacks.trigger(f'before_{self.stage}_forward', trainer=self.trainer)
        model_out = self.model(*args, **kwds)
        self.trainer.callbacks.trigger(f'after_{self.stage}_forward', trainer=self.trainer, model_out=model_out)
        return model_out

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def parameters(self):
        return self.model.parameters()

    def modules(self):
        return self.model.modules()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
