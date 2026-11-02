"""
PyTorch分布式与混合精度训练。

直接运行本文件会自动使用可见GPU，无GPU时使用单CPU进程。
在入口设置launch(main, num_processes=4)可指定四进程。
DataLoader的batch_size=32表示每进程32个样本，全局批量随进程数增加。
也可使用torchrun --standalone --nproc-per-node=4 examples/21-distributed.py。
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from deepepochs import Trainer, launch, rename, set_seed
from deepepochs import metrics as dm


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    precision = 'fp16' if device == 'cuda' else 'fp32'
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

    @rename('m.')
    def metrics(preds, targets):
        return {
            'p': dm.precision(preds, targets, average='macro'),
            'r': dm.recall(preds, targets, average='macro')
        }

    trainer = Trainer(model, F.cross_entropy, opt, epochs=2,
                      metrics=[metrics],
                      device=device,
                      precision=precision,
                      )
    try:
        data_dir = './datasets'
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 协调共享数据的下载与缓存准备，各进程随后使用相同文件。
        with trainer.ddp_data():
            mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
            test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

        # 独立生成器让每个进程使用相同的数据划分。
        train_ds, val_ds = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        train_dl = DataLoader(train_ds, batch_size=32)
        val_dl = DataLoader(val_ds, batch_size=32)
        test_dl = DataLoader(test_ds, batch_size=32)
        trainer.fit(train_dl, val_dl)
        trainer.test(test_dl)
    finally:
        trainer.close()


if __name__ == '__main__':
    launch(main)
