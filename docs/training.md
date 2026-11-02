# 设备与训练配置

数据和模型准备好后，可以在构造 `Trainer` 时配置运行设备、计算精度和参数更新方式。训练器默认选择可用的 CUDA 设备，没有 CUDA 时使用 CPU；输入数据会随模型一起放到该设备。通过 `device` 可以显式指定 PyTorch 设备字符串或 `torch.device`，使用 MPS 时需要显式设置该参数。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `device` | 有 CUDA 时使用 CUDA，否则 CPU | 模型与数据所在的设备 |
| `precision` | `fp32` | `fp32`、`fp16` 或 `bf16`；FP16 需要 CUDA |
| `gradient_accumulation_steps` | `1` | 每次参数更新累积的 DataLoader batch 数量 |
| `split_batches` | `False` | DataLoader 的 batch_size 为每进程批量；True 表示按进程数平分全局批量 |
| `ddp_kwargs` | `None` | 多进程训练时传给 PyTorch 分布式数据并行（DDP）的参数字典 |
| `non_blocking` | `False` | 自动搬运输入与标签时传给 Tensor.to 的参数 |
| `compile_model` | `False` | 使用 `torch.compile` 编译模型 |
| `compute_batch_metrics` | `True` | 是否逐批求值损失以外的指标；False 时只在任务结束求值 |

终端默认先显示训练信息，然后每个 batch 在原行更新损失和所有可用指标，每个 epoch 结束后保留一行 `TRAIN> … VAL> …` 摘要。训练信息在初始化和恢复完成后显示实际配置，`init lr` 表示本次 `fit` 开始时各优化器、各参数组的学习率。`accumulate` 仅在 `gradient_accumulation_steps>1` 时显示自动梯度累积步数。轮末摘要在指标后留三个空格，以 `[3.65s]` 形式显示该轮耗时。

`show_info` 控制每次 `fit` 的训练信息与结束概要，`log_batch=False` 仅隐藏批次输出，`log_tqdm=True` 在交互终端显示进度条，轮末摘要仍独立保留。`log_long=True` 使用七位小数，默认四位；学习率和很小的非零指标使用科学计数法。Checkpoint 保存结果在训练结束时按监控器汇总，日志回调共用的实际 TensorBoard 路径统一提示一次。启动、恢复、权重来源、停止及故障提示在相应操作发生时显示。颜色、文件输出及关键事件见[终端输出](logging.md#终端输出)。

显示与指标求值分别配置：`log_batch=False` 不改变指标计算；需要让完整数据指标只在任务结束求值时，设置 `compute_batch_metrics=False`。损失继续逐批计算，批次行显示当前可用的指标。

## 混合精度与梯度累积

单个 batch 的大小常受显存限制。混合精度让部分计算使用 FP16 或 BF16，通常有助于降低显存占用、提高吞吐；梯度累积则把多个小 batch 的梯度合并到一次参数更新中，用于扩大有效批量。两者可以独立使用，也可以一起配置：

```python
with Trainer(
    model,
    F.cross_entropy,
    opt,
    device='cuda',
    precision='fp16',
    gradient_accumulation_steps=2,
) as trainer:
    trainer.fit(train_dl, val_dl)
```

例子中，`precision='fp16'` 为前向计算和损失计算启用 PyTorch 的自动混合精度（autocast）。CUDA FP16 同时使用 GradScaler 缩放梯度，降低小梯度下溢的影响；选择 `bf16` 时，目标设备需要支持相应计算。

`gradient_accumulation_steps=2` 表示每读取两个 DataLoader batch 更新一次参数，这两个 batch 构成一个累积窗口。训练器按窗口中的实际样本数归一化梯度，轮末不足两个 batch 时也会完成更新。分布式训练在需要更新时同步梯度，使各进程使用一致的累计结果。

这里的计数需要与损失均值的分母一致。`fit`、`test` 和 `EpochTask` 的 `batch_size` 参数提供这一统计与归一化计数，DataLoader 的同名参数负责实际批量，两者分别配置。默认 `None` 根据当前输入推断，普通逐样本平均损失可直接使用默认计数。对于使用 `ignore_index=-100` 的 token 任务，若损失按有效 token 数取均值，可以通过计数函数返回该值：

```python
def valid_tokens(batch_x, batch_y):
    return (batch_y != -100).sum().item()

loss = nn.CrossEntropyLoss(ignore_index=-100)
with Trainer(model, loss, opt, gradient_accumulation_steps=2) as trainer:
    trainer.fit(train_dl, batch_size=valid_tokens)
```

这个写法要求每批包含有效 token。使用类别权重时，计数应采用参与损失均值的权重总量。每批损失乘以对应计数后再累积，才能在窗口末按同一口径归一化。按样本计数时，尾批应使用实际样本数；需要显式计数函数时可返回 `batch_y.shape[0]`，保留 `None` 则由训练器自动推断。

训练任务中断时，训练器丢弃未完成窗口的梯度、损失统计和临时缩放状态，保留已经完成的参数更新。再次调用 `fit` 时，从新的累积窗口开始。

如果需要限制梯度大小，可以在 `callbacks` 中加入 `GradClipCallback`，按元素值或范数（norm）裁剪。启用 FP16 时，裁剪在梯度反缩放后执行；多个优化器使用 norm 裁剪时，有限梯度组参与联合范数计算。GradScaler 根据各优化器的溢出结果决定是否更新参数，以及是否推进跟随参数更新的 step 调度器。

## 训练轮数与批量上限

`Trainer(..., epochs=5)` 设置目标总轮数，初次调用 `fit(..., epochs=N)` 可以覆盖这个目标。`fit` 返回本次调用执行的逐轮训练和验证指标。同一个训练器可以继续训练：再次调用 `fit(epochs=n)` 时，从已完成轮数起额外训练 n 轮；省略 `epochs` 则继续到既定目标，达到目标后不再增加轮次。

```python
with Trainer(model, F.cross_entropy, opt, epochs=2) as trainer:
    first = trainer.fit(train_dl, val_dl)
    continued = trainer.fit(train_dl, val_dl, epochs=1)
```

正常完成时，这个例子共训练 3 轮，`first` 包含前两轮记录，`continued` 包含追加的一轮记录。

`fit(train_dl, val_dl, limit_train_batches=10, limit_val_batches=5)` 限制本次调用中每项训练任务和验证任务执行的本地 batch 数，接受正整数或 `None`。`None` 使用各 `EpochTask.max_batches` 的设置；显式上限覆盖本次调用中对应阶段的任务设置，保留任务原配置。累积窗口在实际最后一个 batch 完成更新。

`non_blocking=True` 只控制自动将输入与标签搬到训练设备时的 `Tensor.to` 调用。它不调整模型、Checkpoint、指标归约或预测结果回到 CPU 的方式。关闭 `auto_traindata_to_device` 时，训练数据由自定义步骤自行搬运。

将训练器放在 `with Trainer(...) as trainer:` 中使用，正常结束和异常退出时都会清理资源。自行管理生命周期时，结束使用后调用 `trainer.close()`。

## 优化器与学习率调度

直接传入 PyTorch 优化器即可训练。需要学习率调度时，使用 `Optimizer(opt, scheduler=..., sched_on='epoch', sched_with_loss=False, monitor=None, on_stage='val')` 组合优化器与调度器，再传入 `Trainer`。`sched_on` 可选 `step` 或 `epoch`，分别跟随实际参数更新或每轮结束推进。

`sched_with_loss=True` 用于 `ReduceLROnPlateau` 等需要指标参数的调度器。epoch 调度的 `monitor` 指定完整指标键，`on_stage` 选择 `train` 或 `val`；选择验证阶段时，没有执行验证的轮次跳过该调度。任务有名称时，使用带前缀的键：

```python
from deepepochs import EpochTask, Optimizer, Trainer

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min')
optimizer = Optimizer(opt, scheduler, sched_with_loss=True,
                      monitor='main/loss', on_stage='val')
with Trainer(model, F.cross_entropy, optimizer, epochs=5) as trainer:
    trainer.fit(train_dl, val_tasks=[EpochTask(val_dl, name='main')])
```

`monitor=None` 时，epoch 指标调度优先使用唯一的验证 loss，没有验证 loss 时使用唯一的训练 loss；同一阶段有多个任务 loss 时需要显式指定来源。`monitor`、`on_stage` 只选择 epoch 指标调度的来源；step 调度仍使用各优化器在实际更新窗口中累计的 loss，不读取 epoch 指标。

多优化器各自管理参数更新和 loss 窗口。`LinearCosineScheduler` 提供线性预热与余弦学习率调整；用法见[学习率调度示例](../examples/9-lr-schedule.py)和[多优化器示例](../examples/10-multi-optimizers.py)。

查找学习率时，可以使用 `LRFindCallback` 逐批调整学习率并记录损失。回调自动读取当前训练任务的 `loss`，具名任务读取 `任务名/loss`；跨任务、跨 epoch 连续计数。`compute_batch_metrics=False` 时批次损失仍可用于查找，完整用法见[学习率查找示例](../examples/5-lr-find.py)。

---

[返回首页](../README.md) · [数据与指标](data-and-metrics.md) · [分布式训练](distributed.md) · [Checkpoint](checkpoints.md)
