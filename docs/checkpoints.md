# Checkpoint 与恢复训练

[README](../README.md) · [数据与预测](data-and-metrics.md) · [分布式训练](distributed.md)

Checkpoint 保存一个 epoch 完成后的训练状态，包含模型、优化器、调度器、早停状态，以及启用的 AMP GradScaler 状态。恢复后从下一轮继续训练。

## 保存与早停

`CheckCallback` 根据训练或验证指标保存快照。`monitor` 使用汇总结果中的完整指标名；具名 `EpochTask` 的指标带有任务名前缀，例如 `validation/loss`。`on_stage='val'` 监控验证，`on_stage='train'` 监控训练，`mode='min'` 或 `'max'` 指定改善方向。

| 参数 | 行为 |
| --- | --- |
| `save_best=True` | 指标严格优于历史最佳值时保存 |
| `save_best=False` | 每次监控都保存，保留最近一次监控时的状态 |
| `patience=0` | 关闭早停 |
| `patience=N` | 连续 N 次监控未达到早停改善阈值时停止 |
| `min_delta=0` | 任意严格改善都会重置早停计数 |
| `min_delta>0` | 相对早停基准的绝对改善量必须超过该值，才重置早停计数 |

`best_value` 跟踪实际最佳指标，`early_stop_baseline` 跟踪最近一次达到 `min_delta` 要求的指标。两者分别更新：很小的改善仍会保存最佳权重，同时可能增加早停计数。例如 `mode='min', min_delta=0.01` 时，指标从 `1.000` 到 `0.995` 会更新最佳值并保存；到 `0.989` 时，相对 `1.000` 的累计改善超过 `0.01`，才更新早停基准并清零计数。验证被 `val_freq` 跳过的轮次不参与验证监控计数。

沿用[基础示例](../README.md#基本使用)中准备的模型、优化器和训练、验证 DataLoader，并用 `test_dl` 表示独立测试集。按验证损失保存，并在连续 5 次监控未达到阈值时停止：

```python
from torch.nn import functional as F
from deepepochs import CheckCallback, Trainer

checker = CheckCallback('loss', on_stage='val', mode='min',
                        patience=5, min_delta=1e-3)
with Trainer(model, F.cross_entropy, opt, epochs=5,
             callbacks=[checker], resume=True) as trainer:
    trainer.fit(train_dl, val_dl)
    results = trainer.test(test_dl, weights=checker)
```

测试显式选取 `checker` 按验证损失保存的最佳模型。

`running_id` 命名本次运行的日志与 Checkpoint 子目录。只有一个监控器时，文件名为 `checkpoint.ckpt`；多个监控器按声明顺序分别保存 `checkpoint-1.ckpt`、`checkpoint-2.ckpt`。每个快照都包含本轮全部监控器的状态。`after_epoch` 回调请求停止时，训练器会完成本轮已登记的保存请求。轮末保存不另起提示行；训练结束时，各监控器分别报告本次最后成功保存的快照，包含保存时的指标、轮次和路径。本次没有保存但引用已有文件时，使用中性的 `CHECKPOINT` 提示列出监控键与路径。

## 选择恢复入口

`Trainer(..., resume=True)` 汇总各监控器 `ckpt_dir` 中最近运行的候选文件，再按文件修改时间选择最近的 Checkpoint。目录中没有 Checkpoint 时从头训练。`resume='运行ID'` 只查找各目录中的该次运行，指定运行缺失或文件无法读取时会报告错误。自动恢复在初始化回调完成后应用快照中的模型和优化器等训练状态，随后从下一轮开始。

已知文件路径时可显式加载：

```python
trainer.load_checkpoint('logs/run-1/checkpoint.ckpt')
opt.param_groups[0]['lr'] = 1e-4
trainer.fit(train_dl, val_dl)
```

`load_checkpoint` 立即恢复模型、优化器、调度器、GradScaler 和已完成轮次。下一次 `fit` 在初始化回调前保留调用时的这些状态，并在初始化完成后应用；加载后手动调整的模型参数、buffer 或优化器学习率也随之保留。例子中的后续训练使用学习率 `1e-4`。回调、实际 DataLoader 与随机数状态在初始化后按 epoch 快照恢复。

恢复后的初次 `fit` 仍以 `Trainer(..., epochs=...)` 或本次 `fit(..., epochs=...)` 设置的总轮数为目标。已经调用过 `fit` 后，`fit(..., epochs=N)` 表示额外训练 N 轮；不传 `epochs` 则延续既定目标。

## 测试与预测选择权重

`test(..., weights='auto', limit_batches=None)` 默认选取监控器保存的权重；`predict(loader, weights='current', limit_batches=None)` 默认使用当前模型。两者接受相同的 `weights` 值：

| `weights` | 选择方式 |
| --- | --- |
| `'current'` | 使用调用时模型中的参数与 buffer |
| `'auto'` | 按声明顺序选择首个 `ckpt_path` 非空的 `CheckCallback`；没有这样的监控器时使用当前模型 |
| `CheckCallback` 实例 | 使用这个监控器的快照；尚未保存快照时报告错误 |
| 文件路径（字符串或 `Path`） | 使用指定 Checkpoint 的模型参数与 buffer |

`'auto'` 使用所选监控器当前 `ckpt_path` 指向的文件。`save_best` 决定后续的保存规则；恢复训练后即使调整该设置，已有文件的选择仍以 `ckpt_path` 为准。多个监控器需要明确选择时，直接传入实例或文件路径：

```python
current_results = trainer.test(test_dl, weights='current', limit_batches=10)
selected_results = trainer.test(test_dl, weights=checker)
file_results = trainer.test(test_dl, weights='logs/run-1/checkpoint.ckpt')
```

从快照评估只临时替换模型权重。`test` 完成或异常退出后恢复调用前的参数与 buffer，优化器、调度器和训练轮次保持原状态。

`predict` 逐批产生独立的 CPU 结果，结束迭代时恢复临时权重与调用前的模型模式。提前结束迭代时使用 `contextlib.closing` 关闭生成器，及时完成恢复：

```python
from contextlib import closing

with closing(trainer.predict(input_dl, weights=checker, limit_batches=10)) as predictions:
    for batch_output in predictions:
        consume(batch_output)
        break
```

预测的整批输入、自定义 `predict_step` 和分布式结果规则见[数据与预测](data-and-metrics.md)。

## 快照内容

文件采用 `torch.save` 格式，可用 `torch.load(path, map_location='cpu', weights_only=False)` 读取。

| 字段 | 内容 |
| --- | --- |
| `model_state` | 模型参数与 buffer |
| `opt_state` | 各优化器和调度器的状态 |
| `epoch` | 已完成轮次的零基索引，下一轮从 `epoch + 1` 开始 |
| `checkpoint_file` | 当前 Checkpoint 文件名 |
| `callback_states` | 各监控器的声明序号、指标、阶段、最佳值、早停基准、未改善次数和相对文件引用 |
| `scaler_state` | 启用 GradScaler 时的缩放状态 |
| `world_size` | 保存时的训练进程数 |
| `rank_states` | 每个 rank 的随机数、回调和实际训练/验证 DataLoader 状态 |

回调通过 `state_dict()` / `load_state_dict(state)` 保存与恢复自己的状态，例如 `AnalyzeCallback` 的 `global_step`。匹配依据是回调类型与声明顺序。DataLoader 状态来自实际执行的训练、验证 loader，包含 loader、sampler、batch sampler 上的 `torch.Generator`，以及内置分布式 sampler 的 seed。

## 恢复范围

相同训练配置、任务及回调声明顺序、进程数下，epoch 边界恢复每个 rank 的 Python、NumPy、PyTorch CPU 随机数状态、当前 CUDA 设备的随机数状态，以及回调和 DataLoader 状态。这样可衔接下一轮的数据打乱与随机计算。确定性计算还取决于所用算子和运行环境。

快照不保存 epoch 中途的批次游标，也不保存 `persistent_workers=True` 时存活的数据加载进程及其内部随机状态。

进程数变化后仍可加载模型、优化器、调度器与训练轮次，回调使用保存时 rank 0 的状态；每 rank 的随机数与 DataLoader 状态仅在进程数相同时恢复。

单进程、DDP 和编译模型使用相同的模型参数格式。单个 Checkpoint 可以移动或改名后加载；多个监控器的文件应按其相对目录关系一起搬迁，以便各监控器找到对应文件。

回调通过 `state_dict()` 显式提供需要续训的数值状态。Tensor 状态保存为独立 CPU 数据；`load_state_dict()` 应将数据复制到已有状态 Tensor，或放到回调需要的设备。writer、hooks 和其他运行资源在初始化时重新建立。
