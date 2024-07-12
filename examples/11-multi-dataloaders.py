"""
在训练、验证、测试中使用多个dataloader
"""
from deepepochs import Trainer, EpochTask, metrics as dm
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

def acc(preds, targets):
    return dm.accuracy(preds=preds, targets=targets)

def p(preds, targets):
    return dm.precision(preds=preds, targets=targets)

def r(preds, targets):
    return dm.recall(preds=preds, targets=targets)

opt = torch.optim.Adam(model.parameters(), lr=2e-4)
trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=6)


# 1. 每个dataloader被封装为一个EpochTask，其中会对dataloader中的每个mini-batch进行迭代
# 2. 用于训练、验证和测试的EpochTask是相同的，由训练器自动区分
train_tasks = [
    EpochTask(
        dataloader=train_dl,    # dataloader
        metrics=[acc],          # 当前dataloader上使用指标
        do_loss=True,           # 当前dataloader上是否计算损失
        step_args={}            # 其他传给每个step的参数字典
        ),
    EpochTask(train_dl, metrics=[p])
    ]
val_tasks = [EpochTask(val_dl, metrics=[r]), EpochTask(val_dl, metrics=[acc])]
test_tasks = [EpochTask(test_dl, metrics=[acc, r]), EpochTask(test_dl, metrics=[p])]

# 3. train_tasks参数可以与train_dl参数共同使用，val_tasks也可以用val_dl共同使用
# 4. 实际上，在fit方法会将train_dl和val_dl封装为EpochTask
trainer.fit(train_tasks=train_tasks, val_tasks=val_tasks)
# 5. 同样，tasks参数也可以与test_dl参数共同使用
trainer.test(tasks=test_tasks)
