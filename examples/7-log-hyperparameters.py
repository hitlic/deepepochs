"""
利用tensorboard记录与可视化超参数
"""
from deepepochs import Trainer, CheckCallback, rename, LogCallback, metrics as dm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from itertools import product


data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

@rename('')
def multi_metrics(preds, targets):
    return {
        'acc': dm.accuracy(preds, targets),
        'p': dm.precision(preds, targets),
        'r': dm.recall(preds, targets)
        }

lr_s = [0.001, 0.01]
dim_s = [32, 64]
dropout_s = [0.1]

for lr, dim, dropout in product(lr_s, dim_s, dropout_s):
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
                    hyper_params={                          # 当前训练使用的超参数
                        'lr': lr,
                        'dim': dim,
                        'dropout': dropout
                        }
                    )

    progress = trainer.fit(train_dl, val_dl)
    test_rst = trainer.test(test_dl)                        # 超参数和测试指标值会在测试之后自动写入日志

logger.run_tensorboard()                                    # 启动tensorboard，在HPARAMS中查看
