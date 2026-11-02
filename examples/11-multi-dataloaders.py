"""
在训练、验证、测试中使用多个dataloader
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from deepepochs import EpochTask, Trainer
from deepepochs import metrics as dm

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

def acc(preds, targets):
    return dm.accuracy(preds=preds, targets=targets)

def p(preds, targets):
    return dm.precision(preds=preds, targets=targets)

def r(preds, targets):
    return dm.recall(preds=preds, targets=targets)

opt = torch.optim.Adam(model.parameters(), lr=2e-4)


# 1. 每个dataloader被封装为一个EpochTask，其中会对dataloader中的每个mini-batch进行迭代
# 2. 用于训练、验证和测试的EpochTask是相同的，由训练器自动区分
train_tasks = [
    EpochTask(
        name='T1',
        dataloader=train_dl,    # dataloader
        metrics=[acc],          # 当前dataloader上使用指标
        do_loss=True,           # 验证和测试时是否计算损失；训练始终计算
        ),
    EpochTask(train_dl, metrics=[p], name='T2')
    ]
val_tasks = [EpochTask(val_dl, metrics=[r], name='T3'),
             EpochTask(val_dl, metrics=[acc], name='T1')]
test_tasks = [EpochTask(test_dl, metrics=[acc, r], name='T1'),
              EpochTask(test_dl, metrics=[p], name='T2')]

# 3. train_dl与train_tasks二选一；val_tasks可以与val_dl共同使用
# 4. 实际上，在fit方法会将train_dl和val_dl封装为EpochTask
with Trainer(model, F.cross_entropy, opt=opt, epochs=6) as trainer:
    trainer.fit(train_tasks=train_tasks, val_tasks=val_tasks)
    # 5. 同样，tasks参数也可以与test_dl参数共同使用
    trainer.test(tasks=test_tasks)
