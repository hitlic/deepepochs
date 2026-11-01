# DeepEpochs

DeepEpochs 是一个简单的 PyTorch 深度学习模型训练工具。将模型、损失函数、优化器和 DataLoader 交给 `Trainer`，即可执行训练、验证和测试。

同时，DeepEpochs 还支持自定义指标、检查点保存与恢复、日志可视化、混合精度与分布式训练等，也可以针对特定任务通过多种方式进行定制。

## 安装

运行环境要求 Python >= 3.8、PyTorch >= 2.0。使用 pip 安装：

```bash
pip install deepepochs
```

源码安装时，进入仓库根目录运行：

```bash
python -m pip install .
```

开发时使用可编辑安装，使源码修改直接用于运行和测试：

```bash
python -m pip install -e .
```

库源码位于 `src/deepepochs`，测试位于 `tests`，示例位于 `examples`。MNIST 示例需要额外安装 `torchvision`，图节点分类示例需要 `DGL`。

## 基本使用

一次基本的训练过程需要准备的关键组成部分包括模型、损失函数、优化器和数据。DeepEpochs 接受这些普通 PyTorch 对象，由 `Trainer` 组织每批数据的前向计算、反向传播和参数更新。

调用 `fit` 开始训练；同时传入验证 DataLoader 时，每轮训练后还会评估验证集。训练结束后，可以调用 `test` 评估模型。这个例子使用随机生成的二分类数据，可以直接运行：

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
trainer = Trainer(model, F.cross_entropy, opt, epochs=5)
progress = trainer.fit(train_dl, val_dl)
results = trainer.test(val_dl)

trainer.close()
```

`epochs=5` 设置目标总轮数。`fit` 返回逐轮的训练和验证记录，`test` 返回测试指标字典。例子复用验证集展示测试接口；实际项目中，应将独立测试集的 DataLoader 传给 `test`。

训练器可以连续执行 `fit` 和 `test`。结束使用时调用 `close()`，完成日志写入并释放训练器管理的资源。

## 数据与指标

### 输入与标签

DataLoader 每次产出的 batch 既可能包含模型输入，也可能包含用于计算损失的标签。基本使用中的 batch 为 `(x, y)`，训练器将 `x` 交给模型，再用预测结果和 `y` 计算损失。其他结构按以下规则处理：

- 指定损失函数时，tuple/list batch 的最后一项作为标签，其余项作为模型输入。标签可以包含嵌套的 tuple、list 或 dict。
- 不指定损失函数时，tuple/list batch 的所有项作为模型输入，模型需要返回标量损失或包含 `loss` 的字典。
- dict batch 的全部字段作为关键字参数传给模型，不自动拆出标签。模型可以读取字典中的标签并计算损失；指标需要标签时，可由自定义步骤或模型输出提供。

输入和标签中的 Tensor 默认会被递归搬到训练设备。模型有多个输入时，可以让 DataLoader 返回 `(x1, x2, y)`，训练器会调用 `model(x1, x2)`。如果模型需要自己接收标签并计算损失，可参考[字典输入输出示例](examples/24-dict_inputs_outputs.py)。

### 指标计算与汇总

除了损失，通常还需要 accuracy、F1 等指标来评价模型。内置指标按调用时的 Tensor 数值计算，支持复用并原地更新预测、标签或混淆矩阵。通过 `metrics` 传入指标函数或函数列表；函数通常接受 `(preds, targets)`，返回标量或字典。可以使用项目提供的指标，也可以传入符合这一约定的 `torchmetrics.functional` 函数。

训练和评估按 batch 执行，而最终指标需要反映整个数据集。有些指标可以对各批结果加权平均，有些则必须汇总预测或混淆矩阵后重新计算。DeepEpochs 用 Patch 保存和合并这些中间数据，通过 `metric_patch` 选择汇总方式：

| `metric_patch` | 聚合方法 | 适用情况 |
| --- | --- | --- |
| `tensor`（默认） | 保存预测和标签，在完整数据上计算指标 | 通用指标，内存开销随数据量增长 |
| `mean` | 根据实际 batch_size 加权聚合每批均值 | accuracy、逐样本平均损失等可加权平均的指标 |
| `confusion` | 聚合混淆矩阵后计算指标 | 分类 accuracy、precision、recall、F1；指标函数接收混淆矩阵 |

选择时要依据指标的计算方式。例如，两个 batch 的平均损失应按各自样本数加权；precision、recall、F1 则需要依据完整预测或混淆矩阵计算，直接平均各批结果通常得不到数据集指标。默认的 `tensor` 方式保存完整预测和标签；分类任务可以选择 `confusion`，用混淆矩阵汇总状态。

`ConfusionMetrics` 可在自定义步骤中一次计算多个分类指标。如果数据的样本维度不能从 Tensor 形状推断，可通过 `batch_size` 函数提供用于加权的样本数。

## 设备与训练配置

数据和模型准备好后，可以在构造 `Trainer` 时配置运行设备、计算精度和参数更新方式。训练器默认选择可用的 CUDA 设备，没有 CUDA 时使用 CPU；输入数据会随模型一起放到该设备。通过 `device` 可以显式指定 PyTorch 设备字符串或 `torch.device`，使用 MPS 时需要显式设置该参数。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `device` | 有 CUDA 时使用 CUDA，否则 CPU | 模型与数据所在的设备 |
| `precision` | `fp32` | `fp32`、`fp16` 或 `bf16`；FP16 需要 CUDA |
| `gradient_accumulation_steps` | `1` | 每次参数更新累积的 DataLoader batch 数量 |
| `split_batches` | `False` | DataLoader 的 batch_size 为每进程批量；True 表示按进程数平分全局批量 |
| `ddp_kwargs` | `None` | 多进程训练时传给 PyTorch 分布式数据并行（DDP）的参数字典 |
| `compile_model` | `False` | 使用 `torch.compile` 编译模型 |

控制台输出由 `log_batch`、`log_tqdm`、`log_long` 和 `show_info` 配置，分别控制逐批指标、进度条、指标小数位数和训练信息显示。默认显示逐批指标及训练信息，指标采用四位小数。

### 混合精度与梯度累积

单个 batch 的大小常受显存限制。混合精度让部分计算使用 FP16 或 BF16，通常有助于降低显存占用、提高吞吐；梯度累积则把多个小 batch 的梯度合并到一次参数更新中，用于扩大有效批量。两者可以独立使用，也可以一起配置：

```python
trainer = Trainer(
    model,
    F.cross_entropy,
    opt,
    device='cuda',
    precision='fp16',
    gradient_accumulation_steps=2,
)
try:
    trainer.fit(train_dl, val_dl)
finally:
    trainer.close()
```

例子中，`precision='fp16'` 为前向计算和损失计算启用 PyTorch 的自动混合精度（autocast）。CUDA FP16 同时使用 GradScaler 缩放梯度，降低小梯度下溢的影响；选择 `bf16` 时，目标设备需要支持相应计算。

`gradient_accumulation_steps=2` 表示每读取两个 DataLoader batch 更新一次参数，这两个 batch 构成一个累积窗口。训练器按窗口中的实际样本数归一化梯度，轮末不足两个 batch 时也会完成更新。分布式训练在需要更新时同步梯度，使各进程使用一致的累计结果。

训练任务中断时，训练器丢弃未完成窗口的梯度、损失统计和临时缩放状态，保留已经完成的参数更新。再次调用 `fit` 时，从新的累积窗口开始。

如果需要限制梯度大小，可以在 `callbacks` 中加入 `GradClipCallback`，按元素值或范数（norm）裁剪。启用 FP16 时，裁剪在梯度反缩放后执行；多个优化器使用 norm 裁剪时，有限梯度组参与联合范数计算。GradScaler 根据各优化器的溢出结果决定是否更新参数，以及是否推进跟随参数更新的 step 调度器。

## 分布式训练

多 GPU 训练使用 PyTorch 的分布式数据并行（DDP）：每个进程持有一份模型，处理自己的数据分片，并在反向传播时同步梯度。各进程据此更新参数，保持模型一致。

DeepEpochs 通过 `launch` 启动这些进程，训练过程仍由 `Trainer` 执行。模型、损失函数、优化器以及 `fit`、`test` 的用法与单进程训练相同。

### 直接运行 Python 脚本

将训练过程写进文件顶层的无参 `main()`，再通过 `launch(main)` 启动，就可以在 IDE 中点击运行，或直接执行 Python 脚本。启动器根据当前可见的 GPU 数量创建训练进程，每张 GPU 对应一个进程；没有 GPU 时使用单个 CPU 进程。

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
    trainer = Trainer(model, F.cross_entropy, opt, epochs=5)
    try:
        trainer.fit(train_dl, val_dl)
        return trainer.test(val_dl)
    finally:
        trainer.close()


if __name__ == '__main__':
    results = launch(main)
```

各训练进程独立执行 `main()`，因此数据、模型、优化器和 `Trainer` 都在其中创建。子进程需要导入该文件，所以 `main` 定义在文件顶层，启动调用放在 `if __name__ == '__main__':` 中。训练函数可以返回指标字典，`launch` 将主进程的普通 Python 结果返回调用方。

例子采用默认的每进程批量，`batch_size=32` 表示每个进程每批处理 32 个样本，全局批量随进程数增加。

需要指定进程数量时，使用 `launch(main, num_processes=4)`；`num_processes=1` 在当前进程中直接执行，适合单卡运行和调试。若只使用部分 GPU，可通过 `CUDA_VISIBLE_DEVICES` 设置可见设备。

[分布式训练示例](examples/21-distributed.py) 和[梯度累积示例](examples/23-grad_accumulate-2.py) 均提供完整运行入口，在仓库根目录直接执行：

```bash
python examples/21-distributed.py
python examples/23-grad_accumulate-2.py
```

这两个文件也可以在 IDE 中直接运行。在 Notebook 中使用多进程时，需要将训练函数保存到可导入的 Python 模块，再导入并调用 `launch`；只定义在交互单元格或局部作用域中的函数无法供子进程导入。

### 数据分片与批量大小

启动多个进程后，还需要把数据分给各进程，并明确一次参数更新用了多少样本。对于普通 DataLoader，训练器自动划分数据，并在每轮更新采样器的 epoch。训练分片保持各进程等长，验证与测试按实际样本划分。任务结束后，各进程汇总全局指标，依据同一结果进行学习率调度和早停；日志与 Checkpoint 由主进程写入。

批量大小由 `split_batches` 决定。设进程数为 `world_size`，普通 DataLoader 自动分片时，完整累积窗口的名义全局有效批量为：

| `split_batches` | DataLoader 的 `batch_size` 含义 | 全局有效批量 |
| --- | --- | --- |
| `False`（默认） | 每进程批量 | `batch_size × world_size × gradient_accumulation_steps` |
| `True` | 各进程平分的全局批量，需能被进程数整除 | `batch_size × gradient_accumulation_steps` |

例如，四卡运行时，`batch_size=32`、累积步数为 2，默认的全局有效批量为 256；设置 `split_batches=True` 后，每进程每批处理 8 个样本，全局有效批量为 64。调用方提供 `DistributedSampler`、自定义 sampler、batch_sampler 或 DataLoader 子类时，训练器直接使用该 DataLoader，批量和分片方式由调用方配置。

### 共享数据准备

各进程都会执行 `main()`，下载数据或创建缓存时需要安排执行顺序。将准备代码放入 `with trainer.main_process_first():`，主进程先执行，随后其他进程执行相同代码并复用已有缓存。单进程脚本也可以使用这一上下文。

### 外部启动与进程管理

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

## Checkpoint 与恢复训练

训练过程中，通常需要保留指标最佳时的模型，也需要在中断后继续训练。Checkpoint 是为此保存的训练快照，包含模型、优化器、调度器、已完成轮次、早停状态，以及启用的 AMP GradScaler 状态。恢复这些状态后，可以从相应的下一轮继续训练。

`CheckCallback` 通过监控训练或验证指标决定何时保存。默认的 `save_best=True` 保存最佳指标对应的快照，适合评估最佳模型；`save_best=False` 保存最近一次监控时的快照，适合从最近保存的状态续训。`patience` 设置连续多少轮未改善后停止训练，取 0 时关闭早停。

例如，按验证损失越小越好的规则保存模型，连续 5 轮未改善时停止，并在启动时尝试恢复已有 Checkpoint：

```python
from deepepochs import CheckCallback

checker = CheckCallback('loss', on_stage='val', mode='min', patience=5)
trainer = Trainer(model, F.cross_entropy, opt, epochs=5,
                  callbacks=[checker], resume=True)
try:
    trainer.fit(train_dl, val_dl)
    results = trainer.test(test_dl)
finally:
    trainer.close()
```

恢复入口由 `resume` 指定。`resume=True` 汇总全部监控器的 `ckpt_dir`，按文件修改时间选择最近的 Checkpoint；传入运行 ID 时，仅在各目录的该次运行中选择。没有 Checkpoint 的首次启动从头训练，指定运行缺失或文件无法读取时会报告错误。也可以调用 `trainer.load_checkpoint(path)` 显式加载指定文件。恢复后的 `epochs` 仍表示要达到的总轮数。

每次运行通过 `running_id` 命名日志和 Checkpoint 子目录。多个 `CheckCallback` 可以分别监控不同指标，并各自保存独立文件。每个快照都包含本轮全部监控器的状态；`after_epoch` 回调请求停止时，训练器会完成本轮已登记的保存请求。

评估时，`test` 按声明顺序选择第一个已有 Checkpoint 的监控器，临时载入其保存的模型权重。使用默认的 `save_best=True` 时，评估的就是该监控器选出的最佳模型。测试完成或异常退出后，模型恢复测试前的权重，训练进度保持不变，可以继续训练。

需要直接查看文件内容时，可以按 `torch.save` 格式读取。主要字段为：

| 字段 | 内容 |
| --- | --- |
| `model_state` | 模型参数与 buffer |
| `opt_state` | 各优化器和调度器的状态 |
| `epoch` | 已完成轮次的零基索引 |
| `checkpoint_file` | 当前 Checkpoint 文件名 |
| `callback_states` | 监控器的声明序号、指标、阶段、最佳值、未改善次数和相对文件引用 |
| `scaler_state` | 启用 GradScaler 时的缩放状态 |

单进程、DDP 和编译模型使用相同的参数格式。单个 Checkpoint 可以移动或改名后加载；多个监控器的文件应按其相对目录关系一起搬迁，以便各监控器找到对应文件。

## 定制训练流程

默认的 `Trainer` 已包含模型调用、损失计算和参数更新。需要在特定时机执行操作时，可以使用 Callback；需要改变每批计算方式时，可以定制步骤；需要特殊指标汇总方法时，可以定制 Patch。

### Callback：在训练事件中执行操作

初始化模型参数、记录每轮结果、检查停止条件，都需要在训练的特定时机执行。`Callback` 提供这些事件入口：继承它并实现需要的方法，再将实例传给 `Trainer(callbacks=...)`。例如，`on_before_fit` 在训练开始前执行，`on_after_epoch` 在每轮结束后执行，[自定义回调示例](examples/14-costomize-callback.py) 展示了具体写法。

分布式训练会在初始化回调完成后同步模型参数与 buffer，使各进程从相同模型状态开始。

### TrainerBase 与 EpochTask：定义每批计算

一个 batch 需要多次调用模型，或者训练、验证、测试需要不同的计算方式时，可以自定义步骤方法。继承 `TrainerBase` 可以为训练器定义步骤。

如果多个数据集需要分别执行，`EpochTask` 可以把一个 DataLoader、指标和步骤参数组织为一项训练、验证或测试任务。继承它还可以为不同任务定义各自的步骤。

步骤方法包括通用的 `step`、阶段专用的 `train_step`、`val_step`、`test_step`，以及验证与测试共用的 `evaluate_step`。它们接受 `(batch_x, batch_y, **step_args)`，返回 None 或 `{指标名: Patch对象}`。单个输入通过 `self.model(batch_x)` 调用模型；多个输入使用 `self.model(*batch_x)`，字典输入使用 `self.model(**batch_x)`。

训练器先选择 `step`，再选择相应的阶段专用方法；验证和测试未定义阶段专用方法时，使用 `evaluate_step`。查找同名方法时，优先采用 `EpochTask` 的实现，再采用训练器的实现。创建 `EpochTask` 时提供的额外关键字参数会传给步骤方法。

一个阶段可以安排多个 `EpochTask`，各任务分别计算自己的指标。使用任务列表时，`train_dl` 与 `train_tasks` 二选一，`val_dl` 可以与 `val_tasks` 合并，`test_dl` 可以与 `tasks` 合并。合并在本次调用的独立列表中完成，传入的任务列表可在多次调用中复用。[自定义训练器示例](examples/15-customize-steps-1.py) 与[自定义任务示例](examples/16-customize-steps-2.py) 分别展示了两种组织方式。

自定义训练步骤还需要决定何时反向传播和更新参数。`self.loss(...)` 默认完成损失计算、反向传播和优化；设置 `do_optimize=False` 时，先计算损失并反向传播，再由 `self.optimize(optimize_index=...)` 选择优化器、触发更新，适用于多优化器任务和手动梯度累积。

手动切分一个 DataLoader batch 时，每个子 batch 调用 `self.loss(..., loss_adjust=子批样本数 / 整批样本数, do_optimize=False)`，最后调用一次 `self.optimize()`。非末子批的前向和反向可以一起放在 `self.no_sync()` 中，单进程下也可使用这一上下文。[手动子批示例](examples/22-grad_accumulate-1.py) 展示了完整循环，[跨批累积示例](examples/23-grad_accumulate-2.py) 使用 `gradient_accumulation_steps` 管理 DataLoader batch 之间的累积。

### Patch：定义指标汇总方式

步骤方法按 batch 返回结果，任务结束后还需要把这些结果合成整个数据集的指标。返回 Patch 对象，就能将用于汇总的数据及其计算方法一并交给训练器。例如，`TensorPatch` 保存预测和标签，`ValuePatch` 保存已计算的批均值与样本数。`MeanPatch` 与 `ValuePatch` 保存的统计值与训练计算图分离。

需要特殊汇总方式时，继承 `PatchBase` 并实现 `load`、`add` 和 `forward`。`load` 装载当前 batch 数据并返回自身，`add` 合并状态，`forward` 返回最终指标或字典。[自定义 Patch 示例](examples/20-customize-patch.py) 展示了完整实现。

图示为 tuple/list batch 中输入与标签分开的数据流。`EpochTask` 从 DataLoader 读取 batch，调用模型获得预测，再将损失和指标装入对应 Patch。任务结束时，各 Patch 计算汇总结果，形成阶段指标字典。

<img src="imgs/data_flow.png" width="80%" alt="DeepEpochs数据流图"/>

## 日志与预测解释

观察训练时，可以从指标曲线了解整体变化，再查看具体样本或模型内部计算，定位需要关注的部分。日志和预测解释均通过回调接入训练器。

### 训练曲线与超参数

将 `LogCallback(log_dir='./logs')` 加入 `callbacks`，即可把指标和学习率写入 TensorBoard。运行 `tensorboard --logdir=./logs` 后，可以在浏览器中查看曲线。设置 `log_graph=True` 还可以记录模型结构图，`example_input` 用于指定结构图的输入，未指定时采用第一个训练 batch。

比较多组实验时，通过 `Trainer` 的 `hyper_params` 传入超参数字典，测试完成后可将超参数与测试指标一起记录。

### 查看预测样本

总体指标不能直接说明哪些样本预测得好、哪些样本容易出错。`InterpreteCallback` 按逐样本指标保留 `k` 条记录，`mode='max'` 选择分数最高的样本，`mode='min'` 选择最低的样本；传入的 `metric` 需要返回逐样本分数。

通过 `top_samples()` 取得样本列表，每条记录为 `[指标值, 预测, 标签, 输入特征]`。单输入特征为 Tensor，多输入为 tuple 或 dict；输入、预测和标签保存为各样本独立的 CPU Tensor。设置 `image_data=True` 可以记录图像，多输入时采用首个输入，字典按字段插入顺序选择。

### 观察层输出与梯度

需要进一步检查模型内部计算时，`AnalyzeCallback` 可以记录各层的前向输出或反向梯度，并在训练结束、异常退出或训练器关闭时移除记录所用的 hooks。

各日志回调共享一个由训练器管理的 TensorBoard writer，`fit` 后的 `test` 会继续写入同一日志。结束使用时调用 `close()`，完成待写日志的保存并退出写线程。

## 示例

仓库中的 24 个示例覆盖完整训练和各扩展入口。安装项目后，可以根据需要运行对应文件；MNIST 示例需要 `torchvision`，图节点分类示例需要 DGL。

| 序号 | 功能 | 代码 |
| --- | --- | --- |
| 1 | 基本使用 | [1-basic.py](examples/1-basic.py) |
| 2 | Trainer、fit、test 参数 | [2-basic-params.py](examples/2-basic-params.py) |
| 3 | 性能评价指标 | [3-metrics.py](examples/3-metrics.py) |
| 4 | Checkpoint 与 EarlyStop | [4-checkpoint-earlystop.py](examples/4-checkpoint-earlystop.py) |
| 5 | 学习率查找 | [5-lr-find.py](examples/5-lr-find.py) |
| 6 | TensorBoard 日志 | [6-logger.py](examples/6-logger.py) |
| 7 | 超参数记录 | [7-log-hyperparameters.py](examples/7-log-hyperparameters.py) |
| 8 | 模型预测解释 | [8-interprete.py](examples/8-interprete.py) |
| 9 | 学习率调度 | [9-lr-schedule.py](examples/9-lr-schedule.py) |
| 10 | 多优化器 | [10-multi-optimizers.py](examples/10-multi-optimizers.py) |
| 11 | 多 DataLoader | [11-multi-dataloaders.py](examples/11-multi-dataloaders.py) |
| 12 | 图节点分类 | [12-node-classification.py](examples/12-node-classification.py) |
| 13 | 前向输出与梯度可视化 | [13-weight-grad-visualize.py](examples/13-weight-grad-visualize.py) |
| 14 | 自定义 Callback | [14-costomize-callback.py](examples/14-costomize-callback.py) |
| 15 | 自定义 Trainer 步骤 | [15-customize-steps-1.py](examples/15-customize-steps-1.py) |
| 16 | 自定义 EpochTask 阶段步骤 | [16-customize-steps-2.py](examples/16-customize-steps-2.py) |
| 17 | 共用 EpochTask.step | [17-costomize-steps-3.py](examples/17-costomize-steps-3.py) |
| 18 | ConfusionPatch | [18-use_patches-1.py](examples/18-use_patches-1.py) |
| 19 | 内置 Patch | [19-use_patches-2.py](examples/19-use_patches-2.py) |
| 20 | 自定义 Patch | [20-customize-patch.py](examples/20-customize-patch.py) |
| 21 | Python 入口、分布式与混合精度 | [21-distributed.py](examples/21-distributed.py) |
| 22 | 手动子 batch 梯度累积 | [22-grad_accumulate-1.py](examples/22-grad_accumulate-1.py) |
| 23 | Python 入口与跨 batch 梯度累积 | [23-grad_accumulate-2.py](examples/23-grad_accumulate-2.py) |
| 24 | 字典输入与输出 | [24-dict_inputs_outputs.py](examples/24-dict_inputs_outputs.py) |

## 验证与开发

功能验证使用小数据任务检查训练结果、状态恢复和资源释放；吞吐测量则使用固定模型与输入配置，观察不同精度、不同卡数下的处理速度。开发和复现时，可以使用仓库中的测试与联调脚本。

### 环境与回归结果

回归测试覆盖 CPU 训练、自动混合精度（AMP）、多进程通信和发行包安装。CPU 回归结果与 GPU 实测记录为：

| 环境 | 验证范围 | 结果 |
| --- | --- | --- |
| macOS，Conda `deepepochs`，Python 3.10、PyTorch 2.1.0 | CPU / Gloo | 139 项：116 项通过，23 项跳过（19 项 CUDA、4 项 CPU GradScaler） |
| macOS，Conda `torch`，Python 3.12、PyTorch 2.8.0 | 指标、任务编排、CPU AMP 与启动入口 | 36 项定向：35 项通过，1 项 CUDA 跳过 |
| 云端 4×RTX 3080，各 10 GiB，Python 3.11.8、PyTorch 2.2.2+cu121、CUDA 12.1、NCCL 2.19.3 | GPU 实测记录 | 130 项：126 项通过、4 项 CPU GradScaler 跳过；18 项 CUDA 全部通过，另 3 项打包测试在本地通过 |
| 云端 RTX 3080，0.7.2 wheel | 单卡指标与三卡 FP16 | 单卡1项、三卡8项全部通过 |

0.7.2 发行包的 GPU 复测覆盖原地更新后的混淆矩阵、预测与标签指标，以及三卡 FP16 的任务列表复用、全局指标、早停和调度；[最终发行包验证记录](test_results/release_0_7_2/summary.json)提供配置和结果。

CPU / Gloo 回归覆盖基础训练、Tensor 原地更新后的指标、任务列表复用、指标与 Patch 汇总、嵌套数据、多输入预测解释、学习率调度、Checkpoint 恢复与搬迁，以及停止和日志资源释放。启动入口还检查真实多进程训练、外部启动、结果返回、异常退出和调用方环境恢复。

CPU AMP 定向回归检查多优化器裁剪、溢出结果复用与非有限梯度组隔离。CUDA 回归将 FP32 更新与参考结果对照，并检查 FP16/BF16、裁剪、尾批累积、多优化器、溢出跳步、AMP 状态恢复和嵌套标签设备搬运。

从仓库根目录安装开发环境、运行测试并构建发行包：

```bash
python -m pip install build
python -m pip install -e .
MPLBACKEND=Agg python -m unittest discover -s tests -v
python -m build
```

小数据联调脚本执行 3 轮训练、全局验证和 Checkpoint 临时加载测试，可以用于检查单进程、CPU 多进程或 CUDA 环境：

```bash
python tests/smoke_training.py --output /tmp/deepepochs-smoke-python
torchrun --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 tests/smoke_training.py --output /tmp/deepepochs-smoke-ddp
python tests/smoke_training.py --device cuda --precision fp16 --output /tmp/deepepochs-smoke-cuda
```

### 多 GPU 联调

多 GPU 联调分别检查启动入口和训练过程：前者确认进程、设备及退出行为，后者将分布式计算与全局参考结果对照，检查训练数值和状态是否一致。

`launch` 在 1/2/3/4 卡与 FP32/FP16/BF16 的组合中完成 12 组 GPU 检查，另检查可见 GPU 子集的自动进程分配和外部 `torchrun` 入口，共 14 组全部通过。检查涵盖设备绑定、全局指标参考、参数和 GradScaler 同步、Checkpoint、日志线程与进程组释放、主进程返回值，以及调用方 OMP 环境和信号处理恢复。

示例 21 和 23 均通过 Python 入口自动使用三卡和四卡，完成两轮训练与测试。本文的完整 `launch` 示例在单进程、三进程 CPU 及三卡 GPU 环境中完成五轮训练并返回测试结果。GPU 启动联调的配置与原始结果保存在[Python 启动入口验证记录](test_results/launch/summary.json)中。

小数据脚本可以直接检查 Python 启动入口。CPU 示例显式指定两个进程，CUDA 示例自动使用可见设备，并检查实际进程数。脚本采用全局批量，三卡示例将其设为 12，由每个进程处理 4 个样本：

```bash
python tests/smoke_launch.py --num-processes=2 --expected-world-size=2 --device=cpu --output /tmp/deepepochs-launch-cpu
python tests/smoke_launch.py --expected-world-size=4 --device=cuda --precision=fp16 --output /tmp/deepepochs-launch-cuda
CUDA_VISIBLE_DEVICES=0,1,2 python tests/smoke_launch.py --expected-world-size=3 --batch-size=12 --device=cuda --precision=fp16 --output /tmp/deepepochs-launch-three
```

在 2/3/4 卡与 FP32/FP16/BF16 的组合中，NCCL 训练共 102 项检查全部通过，包含 18 项中断窗口后重试的检查。重试验证覆盖未同步微批和更新前异常，并与独立参考比较权重、调度损失及各进程状态；配置和结果见[累积窗口与训练验证记录](test_results/accumulation/summary.json)。

评估矩阵的 63 项检查覆盖全局指标、空分片、调度、早停、多监控器目录恢复、Checkpoint 和预测解释。其中三卡训练有 34 项检查、评估有 21 项检查；自动启动、README 和示例的结果见[三卡 GPU 验证记录](test_results/three_gpu/summary.json)。

FP32、FP16 的四卡 Checkpoint 分别通过 4 项单卡加载、搬迁、续训和测试状态恢复检查，FP16 同时恢复 GradScaler 状态。四卡 FP16 的普通模式与 Inductor 编译模式也完成训练联调。各项检查的配置与结果保存在[验证汇总及原始 JSON 记录](test_results/rtx3080/summary.json)中。

可以先生成四卡训练和评估结果，再将评估脚本生成的 Checkpoint 交给单卡脚本继续训练：

```bash
torchrun --standalone --nproc-per-node=4 tests/multigpu_training.py --expected-world-size 4 --precision fp16 --output /tmp/deepepochs-training-4-fp16.json
torchrun --standalone --nproc-per-node=4 tests/multigpu_evaluation.py --expected-world-size 4 --precision fp16 --output /tmp/deepepochs-evaluation-4-fp16
python tests/single_gpu_checkpoint.py --checkpoint /tmp/deepepochs-evaluation-4-fp16/checkpoints/multiple/checkpoint-2.ckpt --source-world-size 4 --precision fp16 --device cuda:0 --output /tmp/deepepochs-single-checkpoint
```

### 吞吐测量

吞吐以每秒处理的图像数衡量。测量使用同一四卡环境，GPU 间通过 PCIe PHB 互联，无 NVLink；脚本以 CPU 缓存的合成图像驱动 Trainer 的完整批次训练路径。模型为 5,244,968 参数的九层卷积网络，输入大小为 224×224，TF32 关闭。

每卡 batch 固定为 32，因此 1/2/4 卡的全局 batch 分别为 32/64/128，即随卡数增加处理的数据量，这种测量称为弱扩展。预热 20 步后测量 200 步，计时窗口包括 CPU 数据整理、数据传入 GPU（H2D）、模型计算、参数更新和 batch 统计。全局吞吐按全部样本数除以最慢进程的测量时间计算，倍率以同精度的单卡结果为基准。

| 精度 | 1卡（张/s） | 2卡（张/s） | 2卡吞吐倍率 | 4卡（张/s） | 4卡吞吐倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1,187.7 | 1,761.3 | 1.48× | 3,017.3 | 2.54× |
| FP16 | 1,348.1 | 2,044.5 | 1.52× | 3,343.2 | 2.48× |
| BF16 | 1,350.5 | 1,880.8 | 1.39× | 3,270.1 | 2.42× |

每轮的指标汇总和日志位于批次计时窗口之外，完整 `fit` 耗时另行记录。JSON 同时包含各进程的每步耗时、吞吐和峰值显存，可用于查看各进程的运行情况。复现测量时，依次运行各精度与卡数组合：

```bash
for precision in fp32 fp16 bf16; do
  for gpus in 1 2 4; do
    torchrun --standalone --nproc-per-node="$gpus" tests/multigpu_benchmark.py --expected-world-size "$gpus" --precision "$precision" --warmup 20 --steps 200 --batch-size 32 --output "/tmp/deepepochs-benchmark-${gpus}-${precision}.json"
  done
done
```

## 适用范围与约束

- DDP 训练在 `drop_last=False` 时会为等长分片补齐样本，训练指标对应实际处理的数据；验证与测试不重复补齐样本。
- 自定义采样器及 DataLoader 子类由调用方保证分布式分片正确。各进程的任务安排、训练步数及调用顺序需要一致。
- `TensorPatch` 保存并传输完整预测与标签。数据量较大时，可加权平均或混淆矩阵类指标适合使用 `MeanPatch` 或 `ConfusionPatch`。
- 吞吐测量采用合成 CPU 缓存数据。实际输入管线中的读取、解码和预处理会影响端到端吞吐。
