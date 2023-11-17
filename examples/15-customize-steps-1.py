"""
通过TrainerBase定制train_step和evaluate_step
"""
from deepepochs import TrainerBase, ValuePatch, TensorPatch, metrics as mm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# 通过Trainer定制训练和验证步
class Trainer(TrainerBase):
    def train_step(self, batch_x, batch_y, **step_args):
        """
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)

        # 记录损失值
        results = {'loss': ValuePatch(loss, len(model_out))}
        # 记录其他指标值
        for m in step_args.get('metrics', list()):
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results

    def evaluate_step(self, batch_x, batch_y, **step_args):
        """
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        model_out = self.model(*batch_x)
        loss = self.loss(model_out, batch_y)

        # 记录损失值
        results = {} if loss is None else  {'loss': ValuePatch(loss, len(model_out))}
        # 记录其他指标值
        for m in step_args.get('metrics', list()):
            results[m.__name__] = TensorPatch(m, model_out, batch_y)
        return results



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


def m_(preds, targets):
    avg = 'micro'
    cmat = mm.confusion_matrix(preds, targets, 10)
    return {
        'acc': mm.accuracy(conf_mat=cmat),
        'p': mm.precision(conf_mat=cmat, average=avg),
        'r': mm.recall(conf_mat=cmat, average=avg),
        'f1': mm.f1(conf_mat=cmat, average=avg),
    }

trainer = Trainer(model, F.cross_entropy, opt, epochs=2, metrics=[m_])  # 训练器
trainer.fit(train_dl, val_dl, do_val_loss=False)                        # 训练、验证
trainer.test(test_dl, do_loss=False)                                    # 测试
