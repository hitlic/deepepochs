# 日志与预测解释

终端显示训练的当前状态，TensorBoard 保存指标曲线和诊断记录，便于结束后回看。观察模型效果时，可以先看整体指标，再查看具体样本或模型内部计算，定位需要关注的部分。TensorBoard 日志和预测解释通过回调接入训练器。

## 终端输出

默认输出从训练信息开始：每个 batch 在原位置刷新损失和所有可用指标，每个 epoch 结束后将训练、验证指标保留为一行摘要。默认显示和 `log_tqdm=True` 均保留全部批次指标；超过终端宽度的内容自动折行，刷新时清除上一批占用的全部行。无法取得有效终端列宽时，逐行输出完整批次信息。单行中的 `TRAIN>`、`VAL>`、`TEST>` 区分阶段，具名任务保留完整指标键，便于对应监控配置。轮末摘要例如：

```text
E 1/5  TRAIN> loss: 0.5317  acc: 0.8479   VAL> loss: 0.2609  acc: 0.9173   [12.40s]
```

`show_info=True` 时，每次 `fit` 都在初始化和恢复完成后显示实际训练配置，包括版本、设备与进程数、计算精度、轮次范围、学习率、运行 ID 和可训练参数量。`init lr` 表示本次 `fit` 开始时各优化器、各参数组的学习率。`accumulate` 仅在 `gradient_accumulation_steps>1` 时显示自动梯度累积步数。普通 DataLoader 还会显示每进程批量。训练与测试摘要在指标后留三个空格，以 `[12.40s]` 形式显示耗时。训练摘要记录整轮训练与验证的墙钟耗时，测试摘要记录各任务耗时。

| 参数 | 默认值 | 显示行为 |
| --- | --- | --- |
| `show_info` | `True` | 显示每次训练的配置与结束概要 |
| `log_batch` | `True` | 每批刷新损失、可用指标与进度；False 时隐藏批次行 |
| `log_tqdm` | `False` | 在列宽可用的交互终端显示进度条和全部批次指标，轮末摘要和事件消息独立保留 |
| `log_long` | `False` | 普通指标保留四位小数；True 时保留七位 |

启动信息中的 `init lr` 使用科学计数法，多参数组按 `opt0/group0` 等标识分别列出。很小的非零指标也使用科学计数法，避免显示成零。`compute_batch_metrics=False` 时，批次仍显示损失；评估同时设置 `do_loss=False` 时，批次行保留阶段和进度计数，完整指标在任务结束后显示。

颜色在交互终端自动启用，设置 `NO_COLOR` 或 `TERM=dumb` 时关闭。普通文件、管道和非交互输出使用逐行纯文本，保留批次记录与轮末摘要，继续遵循 `log_batch` 的设置。颜色只突出阶段、事件、耗时和非有限数值：

| 内容 | 颜色 |
| --- | --- |
| `TRAIN`、`VAL`、`TEST` | 青色、黄色、紫色 |
| `BEST`、`LATEST` 等保存成功消息 | 绿色 |
| 警告、`STOP`、`STOPPED` | 黄色 |
| `NaN`、`Inf` | 红色 |
| 摘要行尾的耗时 | 灰色 |

Checkpoint 按监控规则在轮末保存，不另起提示行。训练结束时，每个监控器用一条消息报告本次最后成功保存的快照：`BEST` 表示保存最佳监控值对应的快照，`LATEST` 表示保存最近一次监控时的状态；消息包含保存时的指标、轮次和文件位置。本次没有保存但引用已有文件时，以中性的 `CHECKPOINT` 显示监控键与文件位置。

评估前，使用当前模型时显示 `[WEIGHTS] Using current model weights`；加载快照时显示 `[WEIGHTS] Using checkpoint: <path>`。通过监控器选择时，提示还包含 `monitor: val/loss` 等监控键，并在文件中有保存轮次时显示 `saved epoch: N`。

早停和人工中断显示 `STOP` 原因，结束概要通过 `FINISHED` 或 `STOPPED` 给出已完成轮数与耗时。项目自身的业务消息由主进程输出，异常保留原始堆栈及进程信息。

## 训练曲线与超参数

将 `LogCallback(log_dir='./logs')` 加入 `callbacks`，即可把指标和学习率写入 TensorBoard。运行 `tensorboard --logdir=./logs` 后，可以在浏览器中查看曲线。设置 `log_graph=True` 还可以记录模型结构图，`example_input` 用于指定结构图的输入，未指定时采用第一个训练 batch。结构图支持多输入、字典关键字输入和字典输出；DDP 或编译模型使用原始模型绘图。

绘制结构图会额外执行模型前向。绘图隔离 Python、NumPy 全局随机状态，以及 PyTorch CPU 和当前 CUDA 设备的随机状态；正常结束或绘图失败后均恢复，使训练与续训使用的随机序列保持衔接。结构图保存失败时立即显示 `WARN`，训练继续进行。

比较多组实验时，通过 `Trainer` 的 `hyper_params` 传入超参数字典，测试完成后可将超参数与测试指标一起记录。

## 查看预测样本

总体指标不能直接说明哪些样本预测得好、哪些样本容易出错。`InterpreteCallback` 按逐样本指标保留 `k` 条记录，`mode='max'` 选择分数最高的样本，`mode='min'` 选择最低的样本；传入的 `metric` 需要返回逐样本分数。

通过 `top_samples()` 取得样本列表，每条记录为 `[指标值, 预测, 标签, 输入特征]`。单输入特征为 Tensor，多输入为 tuple 或 dict；输入、预测和标签保存为各样本独立的 CPU Tensor。设置 `image_data=True` 可以记录图像，多输入时采用首个输入，字典按字段插入顺序选择。

## 观察层输出与梯度

需要进一步检查模型内部计算时，`AnalyzeCallback` 可以记录各层的前向输出或反向梯度，并在训练结束、异常退出或训练器关闭时移除记录所用的 hooks。

`AnalyzeCallback(mode='backward', log_dir='./logs', every_n_batches=1, layers=None)` 默认分析有可训练参数的叶子层输出梯度；`mode='forward'` 分析叶子层前向输出。采样从首个训练 batch 开始，每隔 `every_n_batches` 个 batch 再记录一次，间隔需为正整数。例如间隔为 2 时，日志 step 为 1、3、5；未采样的 batch 跳过统计与 Tensor 梯度 hook 注册。

`layers` 接受原始模型 `named_modules()` 中的精确名称列表，使用未加 DDP 或 `torch.compile` 包装的路径。默认 `None` 沿用叶子层规则；明确指定名称可以观察父模块的 Tensor 输出，空字符串选择根模块。tuple 输出沿用首项 Tensor，未知名称会报错。明确选层时日志标签包含模块路径，默认标签保留模块索引：

```python
from deepepochs import AnalyzeCallback

analysis = AnalyzeCallback(mode='forward', every_n_batches=10,
                           layers=['encoder.0', 'classifier'])
```

采样只记录当前 batch 实际经过的层，保留采样时的数据，不受后续原地激活或梯度修改影响。BF16 采样转为 FP32 写入 TensorBoard。`global_step` 按训练 batch 递增，并通过回调状态保存和恢复，使多次 `fit` 与恢复训练保持同一采样相位；writer、hooks、采样数据和结构图状态不进入该状态字典。

各日志回调共享一个由训练器管理的 TensorBoard writer，指标和诊断记录按各回调的批次或轮次安排写入。每次 `fit` 的结束回调完成后，训练器统一显示一条 `[LOG] TensorBoard: <实际目录>`，然后显示结束概要；目录来自共享 writer 实际使用的位置。`fit` 后的 `test` 会继续写入同一日志。结束使用时调用 `close()`，完成待写日志的保存并退出写线程。

---

[返回首页](../README.md) · [训练配置](training.md) · [Checkpoint](checkpoints.md) · [定制训练流程](customization.md)
