"""
分析与解释模型的预测效果
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from deepepochs import Trainer
from deepepochs import InterpreteCallback


# datasets
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds = random_split(mnist_full, [55000, 5000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

# model
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

# interpreter
interpreter = InterpreteCallback(
    metric=nn.CrossEntropyLoss(reduction='none'),         # 未经reduce（取均值）的指标
    k=15,                                                 # 记录预测效果最差的k个样本
    stages=['train', 'val', 'test'],                      # 针对的stage
    image_data=True                                       # 是否图像数据
    )

trainer = Trainer(model, F.cross_entropy, opt, epochs=5,
                  callbacks=[interpreter]                 # 启用interpreter
                  )
trainer.fit(train_dl, val_dl)                             # 训练、验证
trainer.test(test_dl)                                     # 测试

# interpreter.top_samples()                               # 返回top k个数据及相关信息
interpreter.run_tensorboard()
