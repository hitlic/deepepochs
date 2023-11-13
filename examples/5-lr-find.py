"""
检测适当的学习率
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from deepepochs import Trainer, LRFindCallback

data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, _ = random_split(mnist_full, [55000, 5000])
train_dl = DataLoader(train_ds, batch_size=32)

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

lr_finder = LRFindCallback(
    max_batch=100,          # 最大尝试mini-batch数量
    min_lr=0.00001,         # 尝试学习率下限
    max_lr=0.02,            # 尝试学习率上限
    opt_id=None             # 优化器id（如果使用多个优化器需要指定是哪一个）
    )

trainer = Trainer(model, F.cross_entropy, opt, epochs=2, callbacks=[lr_finder])
trainer.fit(train_dl)
