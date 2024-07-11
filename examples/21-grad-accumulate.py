"""
梯度累积：
    每次计算一个mini-batch的一部分，累积多次计算的梯度然后更新。当内存或GPU不足时使用。

利用accelerate运行：
    命令行：
        # 在__main__=='__name__'下调用main函数，然后执行
        accelerate launch --num_processes=2 file_name.py
        或
        accelerate launch --num_processes=2 --mixed_precision fp16 file_name.py
    Notebook：
        # 不要在代码中调用main函数，执行如下代码
        from accelerate import notebook_launcher
        notebook_launcher(main, args=(), num_processes=2, mixed_precision="fp16")
"""
from deepepochs import Trainer, metrics as dm, set_seed
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator

set_seed(1)

def main():
    # datasets
    data_dir = './datasets'
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
        return dm.accuracy(preds, targets)

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # 注意：也可以不使用Accelerator，但多GPU训练时速度可能由于梯度同步而变慢
    device = Accelerator(split_batches=True, gradient_accumulation_steps=2)
    device = 'cuda:0'
    device = 'cpu'

    trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=2,
                      device=device,
                      metrics=[acc],
                      gradient_accumulation_steps=1
                      )

    trainer.fit(train_dl, val_dl)
    trainer.test(test_dl)

if __name__ == '__main__':
    main()
