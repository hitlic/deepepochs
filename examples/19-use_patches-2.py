"""
1. Patch是对一个或多个mini-batch运行结果的封装，用于batch指标计算、epoch指标计算。
2. Patch的重要方法：
    - __init__:  (metric, batch_size, name)      如果手动调用则参数无限制，如果在Trainer的参数则要求参数列表必须包含 (metric, batch_size)
    - forward:   ()                              用于计算或处理封装的数据，返回指标值或指标字典
    - load:      (batch_preds, batch_targets)    装入数据
    - add:       (other_obj)                     与另一对象相加
3. 内置Patch
    - PatchBase
        - patches.TensorPatch:        保存每个mini-batch的模型输出和标签                        可作为Trainer参数，也可用于step方法定制
        - patches.MeanPatch:          保存每个mini-batch的样本指标之和                          可作为Trainer参数，也可用于step方法定制
        - patches.ConfusionPatch:     保存每个mini-batch的混淆矩阵                             可作为Trainer参数，也可用于step方法定制
        - patches.ValuePatch:         保存每个mini-batch中每个样本某个值（如损失）之和            只能用于step定制
        - patches.ConfusionMetrics:   保存每个mini-batch的混淆矩阵，能计算多种基于混淆矩阵的指标    只能用于step定制
4. 使用Patch
    - 实例化对象
    - load方法装入数据（该方法返回对象自身）
    - forward方法计算指标
    - 利用sum对多个对象求和得到新对象，也可用 + 运算符
5. 定制Patch
    - 继承MetricPatch，实现load方法、forward方法和add方法，参数要求参见 2.
"""

from deepepochs import Trainer, ValuePatch, TensorPatch, MeanPatch, ConfusionMetrics, EpochTask, metrics as mm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [55000, 5000, 0])
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
trainer = Trainer(model, F.cross_entropy, opt, epochs=2)


def acc(preds, targets):
    return mm.accuracy(preds=preds, targets=targets)

class MyTask(EpochTask):
    def step(self, batch_x, batch_y, **step_args):
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)

        batch_size = len(model_out)
        results = {}
        if loss is not None:
            results = {'loss1': ValuePatch(batch_size).load(loss.detach())}                         # 1. 利用ValuePatch返回损失值，并命名为loss1

        results['tacc'] = TensorPatch(acc).load(model_out, batch_y)                                 # 2. 利用TensorPatch返回计算accuracy指标的数据
        results['macc'] = MeanPatch(acc, batch_size).load(model_out, batch_y)                       # 3. 利用MeanPatch返回计算accuracy指标的数据
        results['cm'] = ConfusionMetrics(metrics=['accuracy'], name='C.').load(model_out, batch_y)  # 4. 利用ConfusionPatch返回计算accuracy指标的数据
        return results


train_task = MyTask(train_dl)
val_task = MyTask(val_dl)
test_task = MyTask(test_dl)

trainer.fit(train_tasks=train_task, val_tasks=val_task)
trainer.test(tasks=test_task)
