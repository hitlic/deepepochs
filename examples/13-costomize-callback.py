"""
自定义Callback

利用Callback初始化模型参数
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from deepepochs import Trainer, Callback


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

opt = torch.optim.Adam(model.parameters(), lr=2e-4)


# 定制callback
class ModelInitCallback(Callback):
    """
    更多Callback方法参见 deepepochs.Callback 类
    """
    def on_before_fit(self, trainer, epochs):
        for param in trainer.model.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)

trainer = Trainer(model, F.cross_entropy, opt, epochs=2,
                  callbacks=[ModelInitCallback()]           # 启用自定义Callback
                  )
trainer.fit(train_dl, val_dl)
