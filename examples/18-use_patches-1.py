"""
使用ConfusionPatch及相应的模型性能评价指标
"""
from deepepochs import Trainer, rename, metrics as dm, set_seed
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

set_seed(1)

# datasets
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
# train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
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


@rename('')
def multi_metrics(conf_mat):
    """
    当Trainer的metric_patch参数为'confusion'时，指标函数的输入为混淆矩阵。    # ***
    """
    return {
        'acc': dm.accuracy(conf_mat=conf_mat),
        'p': dm.precision(conf_mat=conf_mat, average='macro'),
        'r': dm.recall(conf_mat=conf_mat, average='macro')
        }

opt = torch.optim.Adam(model.parameters(), lr=2e-4)

trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=2,
                  metrics=[multi_metrics],                              # *** 使用输入为混淆矩阵的指标
                  metric_patch='confusion'                              # *** 使用ConfusionPatch
                  )

progress = trainer.fit(train_dl, val_dl)
test_rst = trainer.test(test_dl)
