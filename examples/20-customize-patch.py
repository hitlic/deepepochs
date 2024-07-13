"""
自定义Patch

当需要非常高效地计算特殊指标时，可选择通过定制Patch来实现。

定制Patch需要继承deepepochs.PatchBase类，重写`forward`方法和`add`方法：
    - `forward()`：无参数，返回当前Patch的指标值
    - `add(obj)`：参数`obj`为另一个Patch对象，返回当前Patch对象和`obj`相加得到的新Patch对象
    - `load(preds, targets)`：装载数据，参数为mini-batch的预测和标签
"""
from deepepochs import Trainer, PatchBase, EpochTask, sum_dicts
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np


# 自定义Patch
class HitsCountPatch(PatchBase):
    """
    用于对Hit@n进行计数的Patch
    """
    def __init__(self, at=(1, 2), name=None):
        super().__init__(name)
        self.at = at
        self.hits_count = {f'@{v}': 0 for v in at}

    def load(self, preds, targets):
        _, ids_sorted = preds.sort(dim=1, descending=True)
        tgts_np = targets.cpu().numpy()
        for n in self.at:
            ids_n = ids_sorted[:,:n].cpu().detach().clone().numpy()
            hit = np.in1d(tgts_np, ids_n, assume_unique=True)
            self.hits_count[f'@{n}'] += hit.sum()
        return self

    def forward(self):
        return self.hits_count

    def add(self, obj):
        # new_obj = HitsCountPatch(self.at, self.name)
        new_obj = self
        new_obj.hits_count = sum_dicts([self.hits_count, obj.hits_count])
        return new_obj


class MyTask(EpochTask):
    def step(self, batch_x, batch_y, **step_args):
        """
        在训练、验证和测试中使用了同一step方法。
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        model_out = self.model(*batch_x)

        self.loss(model_out, batch_y)

        results = {}
        results['nhits'] = HitsCountPatch().load(model_out, batch_y)      # 在训练、验证和测试中使用自定义Patch
        return results


data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

channels, width, height = (1, 28, 28)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(channels * width * height, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 10)
)

opt = torch.optim.Adam(model.parameters(), lr=2e-4)
trainer = Trainer(model, F.cross_entropy, opt, epochs=2)  # 训练器

train_task = MyTask(train_dl)
val_task = MyTask(val_dl)
test_task = MyTask(test_dl)

trainer.fit(train_tasks=train_task, val_tasks=val_task)   # 使用Task
trainer.test(tasks=test_task)                             # 使用Task
