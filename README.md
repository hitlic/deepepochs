# DeepEpochs

DeepEpochs 是一个简单的 PyTorch 深度学习模型训练工具。将模型、损失函数、优化器和 DataLoader 交给 `Trainer`，即可执行训练、验证和测试。

同时，DeepEpochs 还支持自定义指标、检查点保存与恢复、日志可视化、混合精度与分布式训练等，也可以针对特定任务通过多种方式进行定制。

## 安装

运行环境要求 Python >= 3.8、PyTorch >= 2.0。请按机器的 CPU/CUDA 环境，参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装合适的 PyTorch，然后安装 DeepEpochs：

```bash
pip install deepepochs
```

源码安装时，进入仓库根目录运行：

```bash
python -m pip install .
```

MNIST 示例需要额外安装 `torchvision`，图节点分类示例需要 DGL。可编辑安装、回归测试和发行包构建见[验证与开发](https://github.com/hitlic/deepepochs/blob/main/docs/development.md)。

## 基本使用

一次基本的训练过程需要准备的关键组成部分包括模型、损失函数、优化器和数据。DeepEpochs 接受这些普通 PyTorch 对象，由 `Trainer` 组织每批数据的前向计算、反向传播和参数更新。

调用 `fit` 开始训练；同时传入验证 DataLoader 时，每轮训练后还会评估验证集。训练结束后，可以调用 `test` 评估模型。将这两个阶段放在 Trainer 的 `with` 上下文中，离开上下文时会自动关闭日志并释放资源。这个例子使用随机生成的二分类数据，可以直接运行：

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from deepepochs import Trainer

x = torch.randn(256, 16)
y = (x[:, 0] > 0).long()
train_dl = DataLoader(TensorDataset(x[:192], y[:192]), batch_size=32, shuffle=True)
val_dl = DataLoader(TensorDataset(x[192:], y[192:]), batch_size=32)

model = nn.Linear(16, 2)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
with Trainer(model, F.cross_entropy, opt, epochs=5) as trainer:
    progress = trainer.fit(train_dl, val_dl)
    results = trainer.test(val_dl)
```

`epochs=5` 设置目标总轮数。`fit` 返回逐轮的训练和验证记录，`test` 返回测试指标字典。例子复用验证集展示测试接口；实际项目中，应将独立测试集的 DataLoader 传给 `test`。

## 主题文档

| 主题 | 内容 |
| --- | --- |
| [数据与指标](https://github.com/hitlic/deepepochs/blob/main/docs/data-and-metrics.md) | 输入与标签、指标汇总、任务命名、测试与逐批预测 |
| [训练配置](https://github.com/hitlic/deepepochs/blob/main/docs/training.md) | 设备、混合精度、梯度累积、批量上限与学习率调度 |
| [分布式训练](https://github.com/hitlic/deepepochs/blob/main/docs/distributed.md) | 完整 Python 启动例、配置透传、数据分片与进程管理 |
| [Checkpoint 与恢复训练](https://github.com/hitlic/deepepochs/blob/main/docs/checkpoints.md) | 监控、早停、评估权重选择与 epoch 边界恢复 |
| [定制训练流程](https://github.com/hitlic/deepepochs/blob/main/docs/customization.md) | Callback、EpochTask、自定义步骤与 Patch |
| [日志与可视化](https://github.com/hitlic/deepepochs/blob/main/docs/logging.md) | TensorBoard、样本解释、层输出与梯度采样 |
| [验证与开发](https://github.com/hitlic/deepepochs/blob/main/docs/development.md) | 开发测试、CPU 自动化、多 GPU 联调与吞吐测量 |

## 示例

仓库中的 24 个示例覆盖完整训练和各扩展入口。安装项目后，可以根据需要运行对应文件；MNIST 示例需要 `torchvision`，图节点分类示例需要 DGL。

| 序号 | 功能 | 代码 |
| --- | --- | --- |
| 1 | 基本使用 | [1-basic.py](https://github.com/hitlic/deepepochs/blob/main/examples/1-basic.py) |
| 2 | Trainer、fit、test 参数 | [2-basic-params.py](https://github.com/hitlic/deepepochs/blob/main/examples/2-basic-params.py) |
| 3 | 性能评价指标 | [3-metrics.py](https://github.com/hitlic/deepepochs/blob/main/examples/3-metrics.py) |
| 4 | Checkpoint 与 EarlyStop | [4-checkpoint-earlystop.py](https://github.com/hitlic/deepepochs/blob/main/examples/4-checkpoint-earlystop.py) |
| 5 | 学习率查找 | [5-lr-find.py](https://github.com/hitlic/deepepochs/blob/main/examples/5-lr-find.py) |
| 6 | TensorBoard 日志 | [6-logger.py](https://github.com/hitlic/deepepochs/blob/main/examples/6-logger.py) |
| 7 | 超参数记录 | [7-log-hyperparameters.py](https://github.com/hitlic/deepepochs/blob/main/examples/7-log-hyperparameters.py) |
| 8 | 模型预测解释 | [8-interprete.py](https://github.com/hitlic/deepepochs/blob/main/examples/8-interprete.py) |
| 9 | 学习率调度 | [9-lr-schedule.py](https://github.com/hitlic/deepepochs/blob/main/examples/9-lr-schedule.py) |
| 10 | 多优化器 | [10-multi-optimizers.py](https://github.com/hitlic/deepepochs/blob/main/examples/10-multi-optimizers.py) |
| 11 | 多 DataLoader | [11-multi-dataloaders.py](https://github.com/hitlic/deepepochs/blob/main/examples/11-multi-dataloaders.py) |
| 12 | 图节点分类 | [12-node-classification.py](https://github.com/hitlic/deepepochs/blob/main/examples/12-node-classification.py) |
| 13 | 前向输出与梯度可视化 | [13-weight-grad-visualize.py](https://github.com/hitlic/deepepochs/blob/main/examples/13-weight-grad-visualize.py) |
| 14 | 自定义 Callback | [14-costomize-callback.py](https://github.com/hitlic/deepepochs/blob/main/examples/14-costomize-callback.py) |
| 15 | 自定义 Trainer 步骤 | [15-customize-steps-1.py](https://github.com/hitlic/deepepochs/blob/main/examples/15-customize-steps-1.py) |
| 16 | 自定义 EpochTask 阶段步骤 | [16-customize-steps-2.py](https://github.com/hitlic/deepepochs/blob/main/examples/16-customize-steps-2.py) |
| 17 | 共用 EpochTask.step | [17-costomize-steps-3.py](https://github.com/hitlic/deepepochs/blob/main/examples/17-costomize-steps-3.py) |
| 18 | ConfusionPatch | [18-use_patches-1.py](https://github.com/hitlic/deepepochs/blob/main/examples/18-use_patches-1.py) |
| 19 | 内置 Patch | [19-use_patches-2.py](https://github.com/hitlic/deepepochs/blob/main/examples/19-use_patches-2.py) |
| 20 | 自定义 Patch | [20-customize-patch.py](https://github.com/hitlic/deepepochs/blob/main/examples/20-customize-patch.py) |
| 21 | Python 入口、分布式与混合精度 | [21-distributed.py](https://github.com/hitlic/deepepochs/blob/main/examples/21-distributed.py) |
| 22 | 手动子 batch 梯度累积 | [22-grad_accumulate-1.py](https://github.com/hitlic/deepepochs/blob/main/examples/22-grad_accumulate-1.py) |
| 23 | Python 入口与跨 batch 梯度累积 | [23-grad_accumulate-2.py](https://github.com/hitlic/deepepochs/blob/main/examples/23-grad_accumulate-2.py) |
| 24 | 字典输入与输出 | [24-dict_inputs_outputs.py](https://github.com/hitlic/deepepochs/blob/main/examples/24-dict_inputs_outputs.py) |
