# DeepEpochs

Pytorch深度学习模型训练工具。

### 安装

```bash
pip install deepepochs
```

### 使用

#### 数据要求

- 训练集、验证集和测试集是`torch.utils.data.Dataloader`对象
- `Dataloaer`所构造的每个mini-batch数据（`collate_fn`返回值）是一个`tuple`或`list`，其中最后一个是标签
  - 如果训练中不需要标签，则需将最后一项置为`None`

#### 指标计算

- 每个指标是一个函数
  - 有两个参数，分别为模型预测和数据标签
  - 返回值为当前mini-batch上的指标值

#### 应用

```python
from deepepochs import Trainer, CheckCallback, rename, EpochTask, LogCallback
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import functional as MF

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
    return MF.accuracy(preds, targets, task='multiclass', num_classes=10)

@rename('')
def multi_metrics(preds, targets):
    return {
        'p': MF.precision(preds, targets, task='multiclass', num_classes=10),
        'r': MF.recall(preds, targets, task='multiclass', num_classes=10)
        }
checker = CheckCallback('loss', on_stage='val', mode='min', patience=2)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

trainer = Trainer(model, F.cross_entropy, opt=opt, epochs=5, callbacks=checker, metrics=[acc])

progress = trainer.fit(train_dl, val_dl, metrics=[multi_metrics])
test_rst = trainer.test(test_dl)
```

### 示例

|序号|功能说明|代码|
| ---- | ---- | ---- |
|1|基本使用|`examples/1-basic.py`|
|2|训练器、fit方法、test方法的常用参数|`examples/2-basic-params.py`|
|3|模型性能评价指标的使用|`examples/3-metrics.py`|
|4|Checkpoint和EarlyStop|`examples/4-checkpoint-earlystop.py`|
|5|检测适当的学习率|`examples/5-lr-find.py`|
|6|利用Tensorboad记录训练过程|`examples/6-logger.py`|
|7|利用tensorboard记录与可视化超参数|`examples/7-log-hyperparameters.py`|
|8|学习率调度|`examples/8-lr-schedule.py`|
|9|使用多个优化器|`examples/9-multi-optimizers.py`|
|10|在训练、验证、测试中使用多个Dataloader|`examples/10-multi-dataloaders.py`|
|11|利用图神经网络对节点进行分类|`examples/11-node-classification.py`|
|12|模型前向输出和梯度的可视化|`examples/12-weight-grad-visualize.py`|
|13|自定义Callback|`examples/13-costomize-callback.py`|
|14|通过`TrainerBase`定制`train_step`和`evaluate_step`|`examples/14-customize-steps-1.py`|
|15|通过`EpochTask`定制`train_step`和`eval_step`和`test_step`|`examples/15-customize-steps-2.py`|
|16|通过`EpochTask`定制`*step`|`examples/16-costomize-steps-3.py`|
|17|内置Patch的使用|`examples/17-patchs.py`|
|18|自定义Patch|`examples/18-customize-patch.py`|

### 定制训练流程

- 方法1:
    - 第1步：继承`deepepochs.Callback`类，定制满足需要的`Callback`
    - 第2步：使用`deepepochs.Trainer`训练模型，将定制的`Callback`对象作为`Trainer`的`callbacks`参数
- 方法2:
    - 第1步：继承`deepepochs.TrainerBase`类，定制满足需要的`Trainer`，实现`step`、`train_step`、`val_step`、`test_step`或`evaluate_step`方法
        - 这些方法有三个参数
            - `batch_x`：     一个mini-batch的模型输入数据
            - `batch_y`：     一个mini-batch的标签
            -  `**step_args`：可变参数字典，包含`do_loss`、`metrics`等参数
        - 返回值为字典
            - key：指标名称
            - value：`deepepochs.PatchBase`子类对象，可用的Patch有
                - `ValuePatch`：    根据每个batch指标均值（提前计算好）和batch_size，累积计算Epoch指标均值
                - `TensorPatch`：   保存每个batch模型预测输出及标签，根据指定指标函数累积计算Epoch指标均值
                - `MeanPatch`：     保存每个batch指标均值，根据指定指标函数累积计算Epoch指标均值
                - `ConfusionPatch`：累积计算基于混淆矩阵的指标
                - 也可以继承`PatchBase`定义新的Patch（存在复杂指标运算的情况下）
                    - `PatchBase.add`方法
                    - `PatchBase.forward`方法
    - 第2步：调用定制`Trainer`训练模型。
- 方法3:
    - 第1步：继承`deepepochs.EpochTask`类，在其中定义`step`、`train_step`、`val_step`、`test_step`或`evaluate_step`方法
        - 它们的定义方式与`Trainer`中的`*step`方法相同
        - `step`方法优先级最高，即可用于训练也可用于验证和测试（定义了`step`方法，其他方法就会失效）
        - `val_step`、`test_step`优先级高于`evaluate_step`方法
        - `EpochTask`中的`*_step`方法优先级高于`Trainer`中的`*_step`方法
    - 第2步：使用新的`EpochTask`任务进行训练
        - 将`EpochTask`对象作为`Trainer.fit`中`train_tasks`和`val_tasks`的参数值，或者`Trainer.test`方法中`tasks`的参数值

### 数据流图

<img src="imgs/data_flow.png" width="60%" alt="https://github.com/hitlic/deepepochs/blob/main/imgs/data_flow.png"/>
