"""
跨DataLoader batch梯度累积。

gradient_accumulation_steps=2表示每两个batch更新一次参数。
直接运行本文件会自动使用可见GPU，无GPU时使用单CPU进程。
入口设置launch(main, num_processes=4)可指定四进程。
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from deepepochs import Trainer, launch, rename, set_seed
from deepepochs import metrics as mm


@rename('')
def metrics(preds, targets):
    avg = 'macro'
    cmat = mm.confusion_matrix(preds, targets, 10)
    return {
        'acc': mm.accuracy(conf_mat=cmat),
        'f1': mm.f1(conf_mat=cmat, average=avg),
    }


def main():
    set_seed(1.0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    with Trainer(model, F.cross_entropy, opt, epochs=2,
                 metrics=metrics, device=device, precision='fp32',
                 gradient_accumulation_steps=2) as trainer:
        data_dir = './datasets'
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 协调共享数据的下载与缓存准备，各进程随后使用相同文件。
        with trainer.ddp_data():
            mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
            test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

        # 独立生成器让每个进程使用相同的数据划分。
        train_ds, val_ds, _ = random_split(
            mnist_full, [5000, 5000, 50000], generator=torch.Generator().manual_seed(1)
        )
        train_dl = DataLoader(train_ds, batch_size=32)
        val_dl = DataLoader(val_ds, batch_size=32)
        test_dl = DataLoader(test_ds, batch_size=32)
        trainer.fit(train_dl, val_dl)
        trainer.test(test_dl, do_loss=False)


if __name__ == '__main__':
    launch(main)
