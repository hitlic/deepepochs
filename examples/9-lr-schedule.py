"""
学习率调度
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from deepepochs import Trainer, Optimizer, LogCallback


data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds = random_split(mnist_full, [55000, 5000])
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

epoch_num =10

opt = torch.optim.Adam(model.parameters(), lr=2e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch_num*len(train_dl), 0.001)
opt = Optimizer(
    opt=opt,                # 优化器
    scheduler=sched,        # 调度器
    sched_on='step',        # 调度时机，取值为step或epoch，默认为epoch
    sched_with_loss=False   # 调度器在调度时是否需要以loss为参数（例如ReduceLROnPlateau）
    )

loger = LogCallback()
trainer = Trainer(model, F.cross_entropy, opt, epochs=epoch_num, callbacks=[loger])
trainer.fit(train_dl, val_dl)
trainer.test(test_dl)

loger.run_tensorboard()
