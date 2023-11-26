"""
训练器、fit方法、test方法的常用参数
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from deepepochs import Trainer


# 1. --- datasets
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds = random_split(mnist_full, [55000, 5000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

# 2. --- model
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

# 3. --- optimizer
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# model = torch.compile(model)  # 在cpu和cuda下有可能显著提速

# 4. --- train
trainer = Trainer(
    model=model,                # Pytorch模型（nn.Module）
    loss=F.cross_entropy,       # 损失函数，            默认直接返回模型预测（要求模型预测输出为损失）
    opt=opt,                    # 优化器（或优化器列表）， 默认使用学习率为0.001的Adam优化器
    epochs=2,                   # 训练迭代次数，         默认取值1000
    device='cpu',               # 加速设备，cpu、cuda 或 mps，默认情况下如果存在GPU或mps设备会自动使用
    long_output=False,          # 输出为长格式（7位小数）还是短格式（4位小数）
    log_batch=True,             # 训练过程是，是否每个mini-batch都输出一次指标值
    metric_patch='tensor',      # 指标累积计算方法，取值为'tensor'或'mean'
                                #   - tensor 保存每个mini-batch的模型预测和标签计算epoch指标，计算和空间开销大但适用范围广
                                #   - mean 保存每个mini-batch指标均的值，计算和空间开销小，但部分指标（如precision, recall等）不适用
    )

trainer.fit(
    train_dl=train_dl,          # 训练Dataloader
    val_dl=val_dl,              # 验证Dataloader，None表示不进行验证
    val_freq=1,                 # 验证频率
    do_val_loss=True            # 是否计算验证损失
    )

trainer.test(
    test_dl=test_dl,            # 测试Dataloader
    do_loss=True                # 是否计算测试损失
    )
