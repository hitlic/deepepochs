"""
使用模型性能评价指标
"""
from deepepochs import Trainer, rename, metrics as dm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


# datasets
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
train_ds, val_ds, _ = random_split(mnist_full, [45000, 15000, 0])
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

# 指标函数1
def acc(preds, targets):
    """
    指标函数的参数为： (模型预测输出, 标签)
            返回值： 当前mini-batch各样本指标的均值
    """
    return dm.accuracy(preds=preds, targets=targets)

# 指标函数2
def recall(preds, targets):
    return dm.recall(preds=preds, targets=targets, average='macro')

# 指标函数3
@rename('')
def multi_metrics(preds, targets):
    """
    1. 指标函数也可以一次返回多个指标值，以字典的形式返回。
    2. 在输出时，指标的名字由指标函数名和字典的键组成。
    3. 可利用rename来改变函数名，或者直接通过指标函数的__name__属性改变函数名。
    """
    return {
        'mp': dm.precision(preds=preds, targets=targets, average='macro'),
        'mr': dm.recall(preds=preds, targets=targets, average='macro')
        }

opt = torch.optim.Adam(model.parameters(), lr=2e-4)

trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=2,
                  metrics=[acc],                             # 1. 在训练、验证和测试中使用的指标
                  )

progress = trainer.fit(train_dl, val_dl,
                        metrics=[multi_metrics],             # 2. 在训练和验证中使用的指标
                        train_metrics=[multi_metrics],       # 3. 仅在训练中使用的指标
                        val_metrics=[multi_metrics],         # 4. 仅在验证中使用的指标
                        # batch_size=32                      # 5. 特殊情况下需指定batch_size或者能计算batch_size的函数，用于准确计算损失和指标
                        batch_size= lambda x, y: x.shape[0]  #    若为函数，则参数为(batch_x, batch_y)，返回batch_size大小
                        )

test_rst = trainer.test(test_dl,
                        metrics=[recall],                    # 6. 仅在测试中使用的指标
                        batch_size=32                        # 同5.
                        )
