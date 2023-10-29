"""
@author: hitlic

Code Snips：
    打印当前函数名：
        from sys import _getframe
        print(_getframe().f_code.co_name)
"""
from deepepochs import Trainer, CheckCallback, rename, EpochTask, LogCallback
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import functional as MF

import random
import numpy as np
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

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



def acc(preds, targets):
    return MF.accuracy(preds, targets, task='multiclass', num_classes=10)

@rename('')
def multi_metrics(preds, targets):
    return {
        'p': MF.precision(preds, targets, task='multiclass', num_classes=10),
        'r': MF.recall(preds, targets, task='multiclass', num_classes=10)
        }


from itertools import product

lr_s = [0.001, 0.01]
dim_s = [32, 64]
dropout_s = [0.1]

for lr, dim, dropout in product(lr_s, dim_s, dropout_s):
    # 2. --- 模型
    channels, width, height = (1, 28, 28)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * width * height, dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, 10)
    )
    checker = CheckCallback('loss', on_stage='val', mode='min', patience=2)
    logger = LogCallback()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=5,
                    callbacks=[checker, logger],
                    metrics=[multi_metrics],
                    hyper_params={'lr': lr, 'dim': dim, 'dropout': dropout}
                    )

    # 应用示例1：
    progress = trainer.fit(train_dl, val_dl)
    test_rst = trainer.test(test_dl)
