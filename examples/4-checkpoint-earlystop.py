"""
Checkpoint和EarlyStop
"""
from deepepochs import Trainer, CheckCallback, metrics as dm
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

# CheckCallback同时实现了Checkpoint和Early Stopping
checker = CheckCallback(
    monitor='loss',         # 监控指标名称
    on_stage='val',         # 监控阶段，取值为 train 或 val
    mode='min',             # 监督模式，取值为 min 或 max
    patience=2,             # Early Stopping容忍次数，小于1表示不启用Early Stopping
    save_best=False,        # True保存最佳Checkpoint，False保存最新Checkpoint
    ckpt_dir='./logs'       # Checkpoint保存路径
    )

opt = torch.optim.Adam(model.parameters(), lr=2e-4)

trainer = Trainer(model, F.cross_entropy,
                  callbacks=[checker],  # 启用CheckCallback
                  metrics=[acc],
                  resume=True,             # 是否从logs文件平中的Checkpoint加载
                                        #    - False表示不加载
                                        #    - True表示从最新的Checkpoint加载
                                        #    - int、str表示加载相应ID的Checkpoint
                  running_id=1,         # 当前训练的运行编号，用于指定日志和checkpoint的文件夹名，默认使用当前系统时间
                  )

progress = trainer.fit(train_dl, val_dl)
test_rst = trainer.test(test_dl)
