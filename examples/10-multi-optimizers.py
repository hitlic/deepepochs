"""
分别优化特征提取层与分类层：一次反向传播，使用两个优化器更新独立参数。
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from deepepochs import TrainerBase

data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds = random_split(mnist_full, [55000, 5000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


class MultiOptimizerTrainer(TrainerBase):
    def train_step(self, batch_x, batch_y, **step_args):
        model_out = self.model(batch_x)
        # 计算损失并反向传播；do_optimize=False将参数更新留给下面两次调用。
        self.loss(model_out, batch_y, do_optimize=False)
        self.optimize(optimize_index=0)  # 更新特征层，并清空其梯度。
        self.optimize(optimize_index=1)  # 更新分类层，并清空其梯度。


model = Classifier()
# 需要不同优化算法时使用多个优化器；仅学习率不同时可用单个优化器的参数分组。
feature_opt = torch.optim.SGD(model.features.parameters(), lr=0.01, momentum=0.9)
classifier_opt = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

with MultiOptimizerTrainer(model, F.cross_entropy, [feature_opt, classifier_opt], epochs=2) as trainer:
    trainer.fit(train_dl, val_dl)
    trainer.test(test_dl)
