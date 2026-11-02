# 分布式训练

多 GPU 训练使用 PyTorch 的分布式数据并行（DDP）：每个进程持有一份模型，处理自己的数据分片，并在反向传播时同步梯度。各进程据此更新参数，保持模型一致。

DeepEpochs 通过 `launch` 启动这些进程，训练过程仍由 `Trainer` 执行。模型、损失函数、优化器以及 `fit`、`test` 的用法与单进程训练相同。

## 直接运行 Python 脚本

将训练过程写进文件顶层的 `main()`，再通过 `launch(main)` 启动，就可以在 IDE 中点击运行，或直接执行 Python 脚本。启动器根据当前可见的 GPU 数量创建训练进程，每张 GPU 对应一个进程；没有 GPU 时使用单个 CPU 进程。

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from deepepochs import Trainer, launch, set_seed


def main():
    set_seed(42)
    x = torch.randn(256, 16)
    y = (x[:, 0] > 0).long()
    train_dl = DataLoader(TensorDataset(x[:192], y[:192]), batch_size=32, shuffle=True)
    val_dl = DataLoader(TensorDataset(x[192:], y[192:]), batch_size=32)
    model = nn.Linear(16, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    with Trainer(model, F.cross_entropy, opt, epochs=5) as trainer:
        trainer.fit(train_dl, val_dl)
        return trainer.test(val_dl)


if __name__ == '__main__':
    results = launch(main)
```

各训练进程独立执行 `main()`，因此数据、模型、优化器和 `Trainer` 都在其中创建。子进程需要导入该文件，所以 `main` 定义在文件顶层，启动调用放在 `if __name__ == '__main__':` 中。训练函数可以返回指标字典，`launch` 将主进程的普通 Python 结果返回调用方。

例子采用默认的每进程批量，`batch_size=32` 表示每个进程每批处理 32 个样本，全局批量随进程数增加。

需要指定进程数量时，使用 `launch(main, num_processes=4)`；`num_processes=1` 在当前进程中直接执行，适合单卡运行和调试。若只使用部分 GPU，可通过 `CUDA_VISIBLE_DEVICES` 设置可见设备。

[分布式训练示例](../examples/21-distributed.py) 和[梯度累积示例](../examples/23-grad_accumulate-2.py) 均提供完整运行入口，在仓库根目录直接执行：

```bash
python examples/21-distributed.py
python examples/23-grad_accumulate-2.py
```

这两个文件也可以在 IDE 中直接运行。在 Notebook 中使用多进程时，需要将训练函数保存到可导入的 Python 模块，再导入并调用 `launch`；只定义在交互单元格或局部作用域中的函数无法供子进程导入。

## 传入配置

`launch(main, num_processes=None, args=())` 将 `args` 中的位置参数直接传给各进程的训练函数。配置使用字典、列表、数字和字符串等普通可序列化数据，模型、优化器、DataLoader 与 `Trainer` 仍在训练函数内部创建：

```python
from deepepochs import launch


def main(config):
    print(config['learning_rate'])


if __name__ == '__main__':
    config = {'learning_rate': 0.001, 'batch_size': 32}
    launch(main, num_processes=2, args=(config,))
```

第二个位置参数仍表示进程数量；需要传给 `main` 的参数放在 `args` 中。单进程、`torchrun` 环境与已初始化进程组均在当前进程调用 `main(*args)`。多进程入口使用可导入的模块顶层函数。

## 数据分片与批量大小

启动多个进程后，还需要把数据分给各进程，并明确一次参数更新用了多少样本。对于普通 DataLoader，训练器自动划分数据，并在每轮更新采样器的 epoch。训练分片保持各进程等长，验证与测试按实际样本划分。任务结束后，各进程汇总全局指标，依据同一结果进行学习率调度和早停；日志与 Checkpoint 由主进程写入。

批量大小由 `split_batches` 决定。设进程数为 `world_size`，普通 DataLoader 自动分片时，完整累积窗口的名义全局有效批量为：

| `split_batches` | DataLoader 的 `batch_size` 含义 | 全局有效批量 |
| --- | --- | --- |
| `False`（默认） | 每进程批量 | `batch_size × world_size × gradient_accumulation_steps` |
| `True` | 各进程平分的全局批量，需能被进程数整除 | `batch_size × gradient_accumulation_steps` |

例如，四卡运行时，`batch_size=32`、累积步数为 2，默认的全局有效批量为 256；设置 `split_batches=True` 后，每进程每批处理 8 个样本，全局有效批量为 64。调用方提供 `DistributedSampler`、自定义 sampler、batch_sampler 或 DataLoader 子类时，训练器直接使用该 DataLoader，批量和分片方式由调用方配置。

## 共享数据准备

各训练进程都会执行 `main()`，同时向同一目录下载、解压数据或创建缓存可能发生冲突。`Trainer.ddp_data()` 用于协调 DDP 训练中的共享数据准备：主进程先完成准备，其他进程随后执行相同代码并复用已有文件或缓存。

将可能下载数据或创建共享缓存的操作放入 `with trainer.ddp_data():` 即可。路径和数据变换的定义、数据集划分及 DataLoader 创建都放在块外。单进程时直接执行块内代码，使用方式相同。

## 外部启动与进程管理

需要从命令行管理进程时，也可以用 `torchrun` 启动同一份脚本。`launch` 会识别外部启动环境并直接执行训练函数，进程数量由 `torchrun` 决定：

```bash
# 每张GPU一个进程
torchrun --standalone --nproc-per-node=4 examples/21-distributed.py

# CPU双进程
CUDA_VISIBLE_DEVICES="" torchrun --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 examples/21-distributed.py
```

无论使用哪种入口，训练进程都通过 `RANK`、`WORLD_SIZE` 和 `LOCAL_RANK` 取得自己的位置。`Trainer` 据此初始化进程组、选择设备并包装 DDP；CPU 通信使用 Gloo，CUDA 通信使用 NCCL。调用方已初始化进程组时，`launch` 也会直接执行训练函数。

`launch` 使用 PyTorch 启动器，以 `spawn` 创建进程，并使用动态本地端口协调任务。任一训练进程失败时，启动器结束同组进程并将错误返回调用方，任务不自动重启。未设置 `OMP_NUM_THREADS` 时，多进程任务临时采用 `OMP_NUM_THREADS=1`，限制每个进程的 OpenMP 线程数量；任务结束后恢复调用方原有的 OMP 环境和信号处理配置。

进程组由创建方管理。`Trainer.close()` 释放训练器创建的进程组；调用方事先初始化的进程组由调用方关闭。训练器关闭时也会完成日志写入并关闭共享 writer。

## 适用范围与约束

- DDP 训练在 `drop_last=False` 时会为等长分片补齐样本，训练指标对应实际处理的数据；验证与测试不重复补齐样本。
- 自定义采样器及 DataLoader 子类由调用方保证分布式分片正确。各进程的任务安排、训练步数及调用顺序需要一致。
- [逐批预测](data-and-metrics.md)返回各 rank 的本地结果，跨 rank 汇总和全局排序由调用方处理。

---

[返回首页](../README.md) · [训练配置](training.md) · [数据与指标](data-and-metrics.md) · [验证与开发](development.md)
