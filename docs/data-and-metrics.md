# 数据与指标

## 输入与标签

DataLoader 每次产出的 batch 既可能包含模型输入，也可能包含用于计算损失的标签。基本使用中的 batch 为 `(x, y)`，训练器将 `x` 交给模型，再用预测结果和 `y` 计算损失。其他结构按以下规则处理：

- 指定损失函数时，tuple/list batch 的最后一项作为标签，其余项作为模型输入。标签可以包含嵌套的 tuple、list 或 dict。
- 不指定损失函数时，tuple/list batch 的所有项作为模型输入，模型需要返回标量损失或包含 `loss` 的字典。
- dict batch 的全部字段作为关键字参数传给模型，不自动拆出标签。模型可以读取字典中的标签并计算损失；指标需要标签时，可由自定义步骤或模型输出提供。

输入和标签中的 Tensor 默认会被递归搬到训练设备。模型有多个输入时，可以让 DataLoader 返回 `(x1, x2, y)`，训练器会调用 `model(x1, x2)`。如果模型需要自己接收标签并计算损失，可参考[字典输入输出示例](../examples/24-dict_inputs_outputs.py)。

## 指标计算与汇总

除了损失，通常还需要 accuracy、F1 等指标来评价模型。内置指标按调用时的 Tensor 数值计算，支持复用并原地更新预测、标签或混淆矩阵。通过 `metrics` 传入指标函数或函数列表；函数通常接受 `(preds, targets)`，返回标量或字典。可以使用项目提供的指标，也可以传入符合这一约定的 `torchmetrics.functional` 函数。

训练和评估按 batch 执行，而最终指标需要反映整个数据集。有些指标可以对各批结果加权平均，有些则必须汇总预测或混淆矩阵后重新计算。DeepEpochs 用 Patch 保存和合并这些中间数据，通过 `metric_patch` 选择汇总方式：

| `metric_patch` | 聚合方法 | 适用情况 |
| --- | --- | --- |
| `tensor`（默认） | 保存预测和标签，在完整数据上计算指标 | 通用指标，内存开销随数据量增长 |
| `mean` | 按各批统计计数加权累计批均值 | accuracy、逐样本平均损失等可加权平均的指标 |
| `confusion` | 聚合混淆矩阵后计算指标 | 分类 accuracy、precision、recall、F1；指标函数接收混淆矩阵 |

选择时要依据指标的计算方式。例如，两个 batch 的平均损失应按各自样本数加权；precision、recall、F1 则需要依据完整预测或混淆矩阵计算，直接平均各批结果通常得不到数据集指标。默认的 `tensor` 方式保存完整预测和标签；分类任务可以选择 `confusion`，用混淆矩阵汇总状态。

`ConfusionMetrics` 可在自定义步骤中一次计算多个分类指标。

指标默认既在每批结束时求值，也在任务结束时汇总求值。有些指标只对完整数据集有定义。例如，整轮数据包含正负两类，但单个 batch 只有一类时，AUROC 无法逐批计算。此时设置 `Trainer(..., compute_batch_metrics=False)`，让损失以外的指标只在任务结束时求值；`log_batch` 仅控制显示，不改变求值时机。

沿用[基础示例](../README.md#基本使用)的二分类模型与 DataLoader，可以按完整预测计算 AUROC：

```python
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from deepepochs import Trainer


def auroc(preds, targets):
    scores = preds.float().softmax(dim=1)[:, 1].cpu().numpy()
    return roc_auc_score(targets.cpu().numpy(), scores)


with Trainer(model, F.cross_entropy, opt, epochs=5,
             metrics=[auroc], compute_batch_metrics=False) as trainer:
    progress = trainer.fit(train_dl, val_dl)
```

这个例子使用默认的 `tensor` 聚合，任务结束时将完整预测和标签交给 `auroc`。自定义步骤返回的 `TensorPatch` 也遵循这一求值设置。`mean` 聚合需要先计算每批均值，适用于可逐批计算的指标。

关闭逐批指标求值后，批次日志和批次回调仍能取得 `loss`，学习率查找也可继续使用；任务指标保留各自的完整名称。测试任务同时设置 `do_loss=False` 时，批次指标字典为空，任务结束后仍返回汇总指标。

DataLoader 的 `batch_size` 决定每次读取多少样本。`fit`、`test` 和 `EpochTask` 的同名参数用于提供损失与 `mean` 指标的统计计数，同时作为自动梯度累积的归一化计数；它不改变 DataLoader 的实际批量。默认 `batch_size=None` 时，训练器根据当前输入推断计数，普通 Tensor 输入按样本维计算，尾批使用实际样本数。

普通逐样本平均损失通常保留默认值即可。输入结构无法自动推断，或损失按有效 token 等其他计数取均值时，可以传入 `batch_size(batch_x, batch_y)` 函数，返回当前批次的实际计数。固定整数适用于每批计数确实相同的情况；尾批不足时仍指定完整批量，会使损失和 `mean` 指标的加权结果不准确。

## 多任务命名与批量上限

`EpochTask(dataloader, name=None, max_batches=None)` 将数据、指标和步骤参数组织为一项任务。指定 `name='images'` 后，任务返回的 `loss`、`accuracy` 等指标使用 `images/loss`、`images/accuracy` 作为完整键。任务名称需要是非空字符串；多个任务的最终指标键重复时会报错，可为任务指定不同名称。

`max_batches` 是每次执行该任务时的本地 batch 上限，接受正整数或 `None`，`None` 表示完整执行。`fit` 的 `limit_train_batches`、`limit_val_batches` 和 `test` 的 `limit_batches` 可以在本次调用中覆盖各任务的上限；省略时使用各任务自己的 `max_batches`。设置上限后的指标仅覆盖实际执行的样本，分布式时各 rank 按本地分片计数。

## 测试与逐批预测

`trainer.test(test_dl, weights='auto', limit_batches=None)` 计算损失与指标，返回汇总字典。`weights='auto'` 使用声明顺序中首个已有 Checkpoint 的监控器所指权重，没有可用 Checkpoint 时使用当前权重。可以通过 `weights='current'` 评估内存中的模型；选择监控器实例或文件路径的用法见[Checkpoint 与恢复训练](checkpoints.md)。

`trainer.predict(dataloader, weights='current', limit_batches=None)` 返回生成器，每次产出一个 batch 的预测，Tensor 已与计算图分离并复制到 CPU，嵌套 tuple、list、dict 保留结构。它逐批交还结果，调用方可以直接处理每批输出，不必在训练器中保留全部预测。

预测将 DataLoader 产出的整个 batch 作为模型输入：Tensor 直接传入，tuple/list 展开为位置参数，dict 展开为关键字参数。它不会像训练和测试那样拆出末项标签。输入包含标签或样本 ID 时，覆盖 `TrainerBase.predict_step(batch_x)` 自行拆分；预测不经过训练、验证或测试的步骤方法。

```python
from contextlib import closing
from torch.utils.data import DataLoader

input_dl = DataLoader(x, batch_size=32)
with closing(trainer.predict(input_dl, weights='current', limit_batches=10)) as batches:
    for predictions in batches:
        print(predictions)
        break
```

生成器完整消费或关闭时会恢复临时加载前的权重、各子模块的训练模式及当前任务。提前退出循环时使用 `contextlib.closing`，使恢复在离开代码块时完成。

分布式预测沿用评估分片规则，各 rank 返回各自本地的结果，不自动拼接或恢复全局原始顺序。需要汇总并排序时，可将样本 ID 与输入一起传入，在自定义 `predict_step` 中把 ID 和预测一起返回，再由调用方组织结果。

## 适用范围与约束

`TensorPatch` 保存并传输完整预测与标签。数据量较大时，可加权平均的指标使用 `mean`，分类计数类指标使用 `confusion`，减少汇总状态的存储和通信量。

---

[返回首页](../README.md) · [训练配置](training.md) · [定制训练流程](customization.md)
