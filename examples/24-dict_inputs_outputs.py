"""
模型的输入和输出是字典类型的示例，与Hugging Face Transformers的方式兼容。
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


def collate_fn(batch):
    x, y = zip(*batch)
    x = torch.stack(x)
    y = torch.tensor(y)
    return {'x': x, 'y': y}


train_dl = DataLoader(train_ds, batch_size=32, collate_fn=collate_fn)
val_dl = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)


# 2. --- model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        channels, width, height = (1, 28, 28)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 10)
        )

    def forward(self, x, y):
        preds = self.model(x)
        loss = F.cross_entropy(preds, y)
        return {'loss': loss}


model = Model()

# 3. --- optimizer
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# 4. --- train
trainer = Trainer(model, opt=opt, epochs=2)  # 训练器
trainer.fit(train_dl, val_dl)                             # 训练、验证
trainer.test(test_dl)                                     # 测试
