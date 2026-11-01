"""
通过TrainerBase定制train_step，实现累积梯度训练。 
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from deepepochs import TrainerBase, batches, concat, rename, set_seed
from deepepochs import metrics as mm

set_seed(1.0)

data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
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

@rename('')
def metrics(preds, targets):
    avg = 'macro'
    cmat = mm.confusion_matrix(preds, targets, 10)
    return {
        'acc': mm.accuracy(conf_mat=cmat),
        'f1': mm.f1(conf_mat=cmat, average=avg),
    }


class Trainer(TrainerBase):
    def train_step(self, batch_x, batch_y, **step_args):
        """
        定制train_step，实现累积梯度训练。
        将一个DataLoader batch拆成子batch，在全部子batch反向传播后更新一次参数。
        """
        model_out = []      # 保存模型输出
        loss_value = 0.0    # 保存训练损失

        sub_batch_size = 16  # 子batch大小

        # ------ 梯度累积
        for sub_idx, (sub_batch_x, sub_batch_y) in enumerate(zip(batches(batch_x, sub_batch_size), batches(batch_y, sub_batch_size))):
            # 手动将子batch数据放入device
            sub_batch_x, sub_batch_y = sub_batch_x.to(self.device), sub_batch_y.to(self.device)
            loss_adjust = sub_batch_y.shape[0] / batch_y.shape[0]
            is_last = (sub_idx + 1) * sub_batch_size >= batch_y.shape[0]
            # 分布式训练中，仅在最后一个子batch同步梯度。
            with nullcontext() if is_last else self.no_sync():
                sub_model_out = self.model(sub_batch_x)
                loss_value += self.loss(sub_model_out, sub_batch_y, loss_adjust=loss_adjust, do_optimize=False, do_metric=False) * sub_batch_y.shape[0]
            # 保存模型输出
            model_out.append(sub_model_out)

        # ------ 优化及结果处理
        # 优化参数，并梯度清零
        self.optimize()
        # 触发metric处理的callback，确保指标正确计算
        self.do_metric(loss_value/batch_y.shape[0], concat(model_out), batch_y.to(self.device))


trainer = Trainer(model, F.cross_entropy, opt, epochs=2, metrics=metrics,
                  auto_traindata_to_device=False,       # 禁止在训练时自动将数据放入device
                  device='cuda' if torch.cuda.is_available() else 'cpu')
trainer.fit(train_dl, val_dl)                           # 训练、验证
trainer.test(test_dl, do_loss=False)                    # 测试
