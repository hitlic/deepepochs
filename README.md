# DeepEpochs

Pytorch深度学习模型训练工具。

### 安装

```bash
pip install deepepochs
```

### 使用

#### 数据要求

- 训练集、验证集和测试集是`torch.utils.data.Dataloader`对象
- `Dataloaer`中每个mini-batch数据是一个`tuple`或`list`，其中最后一个是标签
  - 如果数据不包含标签，则请将最后一项置为`None`

#### 指标计算

- 每个指标是一个函数
  - 它有两个参数，分别为模型的预测结果和标签
  - 返回值为当前mini-batch上的指标值

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

@rename('')
def multi_metrics(preds, targets):
    return {
        'p': MF.precision(preds, targets, task='multiclass', num_classes=10),
        'r': MF.recall(preds, targets, task='multiclass', num_classes=10)
        }


checker = Checker('loss', mode='min', patience=2)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=100, checker=checker, metrics=[acc, multi_metrics])

progress = trainer.fit(train_dl, val_dl)
test_rst = trainer.test(test_dl)
```

#### 非常规训练流程

- 方法1:
    - 第1步：继承`deepepochs.TrainerBase`类，定制满足需要的`Trainer`，实现`train_step`方法和`evaluate_step`方法
    - 第2步：调用定制`Trainer`训练模型。
- 方法2:
    - 第1步：继承`deepepochs.Callback`类，定制满足需要的Callback
    - 第2步：使用`deepepochs.Learner`训练模型，将定制的Callback作为`Learner`的参数
    - __提示__：`Learner`是具有`Callback`功能的`Trainer`