# deepepochs
Pytorch模型简易训练工具

### 使用

#### 常规训练流程

```python
from deepepochs import Trainer, Checker, rename
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import functional as MF


# datasets
data_dir = './dataset'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

# dataloaders
train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

# pytorch model
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
    return MF.accuracy(preds, targets, task='multiclass', num_classes=10)

@rename('m')
def multi_metrics(preds, targets):
    r =  MF.recall(preds, targets, task='multiclass', num_classes=10)
    f1 = MF.f1_score(preds, targets, task='multiclass', num_classes=10)
    return {'@r': r, '@f1': f1}


checker = Checker('loss', mode='max', patience=2)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=100, checker=checker, metrics=[acc, multi_metrics])

progress = trainer.fit(train_dl, val_dl)
test_rst = trainer.test(test_dl)
```

#### 非常规训练流程

- 第1步：继承`deepepochs.TrainerBase`类，定制满足需要的`Trainer`，实现`train_step`方法和`evaluate_step`方法
- 第2步：调用定制`Trainer`训练模型。
