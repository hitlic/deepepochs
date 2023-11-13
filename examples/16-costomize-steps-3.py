"""
通过EpochTask定制`*step`:
    1. Trainer中的`*step`方法与EpochTask中的`*step`方法作用完全相同，但EpochTask在fit或test中使用时更具灵活性。
    2. `step`方法优先级最高，即可用于训练也可用于验证和测试（定义了`step`方法，其他方法就会失效）
    3. `val_step`、`test_step`优先级高于`evaluate_step`方法
    4. `EpochTask`中的`*step`方法优先级高于`Trainer`中的`*step`方法
"""
from deepepochs import Trainer, ValuePatch, TensorPatch, EpochTask, metrics as mm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


# 定制可用于训练、验证和测试的step方法
class MyTask(EpochTask):
    def step(self, batch_x, batch_y, **step_args):
        """
        在训练、验证和测试中使用了同一step方法。
        注意：本方法返回一个字典，键为指标名，值为封装了数据和指标函数的PatchBase子类对象。
        """
        model_out = self.model(*batch_x)

        loss = None
        if self.stage == 'train':               # 训练（train）step中优化模型
            self.opt.zero_grad()
            loss = self.loss(model_out, batch_y)
            loss.backward()
            self.opt.step()
        elif step_args.get('do_loss', False):  # 验证（val）和测试（test）step中，do_loss为True则计算损失
            loss = self.loss(model_out, batch_y)

        # 记录损失值
        if loss is not None:
            results = {'loss': ValuePatch(loss.detach(), batch_size=len(model_out))}
        else:
            results = {}

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

train_task = MyTask(train_dl)
val_task = MyTask(val_dl)
test_task = MyTask(test_dl)

trainer.fit(train_tasks=train_task, val_tasks=val_task)                 # 使用Task
trainer.test(tasks=test_task)                                           # 使用Task
