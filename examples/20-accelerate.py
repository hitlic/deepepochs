"""
利用HuggingFace accelerate实现多GPU分布式训练、混合精度训练。


1. 定义Accelerator对象（可指定混合精度训练参数）
2. 配置模型、优化器和数据
3. 将Accelerator对象作为Trainer的device参数值，启用Accelerator
4. 代码结构
    - 将训练代码放入一个函数之中，例如`main`
5. 运行
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

# %%writefile file_name.py  # 将Notebook代码Cell写入文件file_name.py

from deepepochs import Trainer, rename, metrics as dm
from accelerate import Accelerator
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


# 1. 定义函数
def main():
    # 2. 定义Accelerator
    accelerator = Accelerator(split_batches=True)

    # 第一个进程先加载数据并缓存，后续进程利用缓存避免每个进程都加载数据
    with accelerator.main_process_first():
        data_dir = './datasets'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
        train_ds, val_ds = random_split(mnist_full, [55000, 5000])
        test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

    train_dl = DataLoader(train_ds, batch_size=32)
    val_dl = DataLoader(val_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)

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

    # 3. 配置模型、优化器和数据
    model, opt, train_dl, val_dl, test_dl = accelerator.prepare(model, opt, train_dl, val_dl, test_dl)

    trainer = Trainer(model, F.cross_entropy, opt, epochs=2,
                      metrics=[metrics],
                      device=accelerator,  # 4. 启用accelerator
                      )
    trainer.fit(train_dl, val_dl)
    trainer.test(test_dl)


# **注意**：在notebook中运行时要注释掉下面两行代码
if __name__=='__main__':
    main()


# 5. 在命令行中运行：
#         accelerate launch --num_processes=2 file_name.py
# 或者
# 5. 在notebook中运行：
#        from accelerate import notebook_launcher
#            notebook_launcher(main, args=(), num_processes=2)
