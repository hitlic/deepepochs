"""
使用多个优化器
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from deepepochs import Trainer, Optimizer


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

# 定义多个优化器，实际使用中每个优化器应针对不同的模型组成部分
# 注意：大多数情况下不需要多个优化器，而是为模型参数分组，每个组使用不同的学习率
opt1 = torch.optim.Adam(model.parameters(), lr=2e-4)
opt2 = torch.optim.Adam(model.parameters(), lr=2e-4)

opts = [opt1, opt2]                         # 第一种方式
opts = [Optimizer(opt1), Optimizer(opt2)]   # 第二种方式，这种方式可为每个优化器指定高度器

trainer = Trainer(model, F.cross_entropy, opts, epochs=2)
trainer.fit(train_dl, val_dl)
trainer.test(test_dl)
