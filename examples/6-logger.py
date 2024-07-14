"""
利用Tensorboad记录训练过程
"""
from deepepochs import Trainer, LogCallback, metrics as dm
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

logger = LogCallback(
    log_dir='./logs',                # 日志保存位置，默认为 ./logs
    log_graph=True,                  # 是否保存模型结构图，默认为False
    # example_input=None             # 保存模型结构图时的输入样例，默认以第一个训练batch_x作为样例输入
    # example_input=train_ds[0][0]   # 指定样例输入
    )  # tensorboard日志callback

opt = torch.optim.Adam(model.parameters(), lr=2e-4)
trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=2, callbacks=[logger])
progress = trainer.fit(train_dl, val_dl, metrics=[acc])

logger.run_tensorboard()            # 启动tensorboard，在GRAPHS中查看模型结构图
