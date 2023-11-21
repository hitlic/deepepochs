"""
内置Patch的使用

1. Patch是数据和指标函数的封装，每个Patch对象封装了一个mini-batch的(preds, targets)数据。
2. Patch对象是可调用对象，执行对象返回指标函数对在mini-batch的(preds, targets)数据上的指标值。
3. Patch对象可以利用`+`运算符将多个Patch合并为一个Patch。
4. DeepEpochs内置了四种Patch对象：
    - `ValuePatch`：    根据每个mini-batch指标均值（提前计算好）和batch_size，累积计算Epoch指标均值
    - `TensorPatch`：   保存每个mini-batch的(preds, targets)，Epoch指标利用所有mini-batch的(preds, targets)数据重新计算
    - `MeanPatch`：     保存每个batch指标均值，Epoch指标值利用每个mini-batch的均值计算
        - 一般`MeanPatch`与`TensorPatch`结果相同，但占用存储空间更小、运算速度更快
        - 不可用于计算'precision', 'recall', 'f1', 'fbeta'等指标
    - `ConfusionPatch`：用于计算基于混淆矩阵的指标，包括'accuracy', 'precision', 'recall', 'f1', 'fbeta'等
5. 在定制训练、验证或测试步时，可以利用这四种Patch作为返回字典的值。
"""

from deepepochs import Trainer, ValuePatch, TensorPatch, MeanPatch, ConfusionPatch, EpochTask, metrics as mm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


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
trainer = Trainer(model, F.cross_entropy, opt, epochs=2)


class MyTask(EpochTask):
    def step(self, batch_x, batch_y, **step_args):
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)

        results = {}
        if loss is not None:
            results = {'loss1': ValuePatch(loss.detach(), batch_size=len(model_out))}         # 1. 利用ValuePatch返回损失值，并命名为loss1

        results['tacc'] = TensorPatch(mm.accuracy, model_out, batch_y)                       # 2. 利用TensorPatch返回计算accuracy指标的数据
        results['macc'] = MeanPatch(mm.accuracy, model_out, batch_y)                         # 3. 利用MeanPatch返回计算accuracy指标的数据
        results['cm'] = ConfusionPatch(model_out, batch_y, metrics=['accuracy'], name='C.')  # 4. 利用ConfusionPatch返回计算accuracy指标的数据
        return results


train_task = MyTask(train_dl)
val_task = MyTask(val_dl)
test_task = MyTask(test_dl)

trainer.fit(train_tasks=train_task, val_tasks=val_task)
trainer.test(tasks=test_task)
