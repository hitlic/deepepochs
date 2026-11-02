# 验证与开发

功能测试使用小数据检查训练更新、指标汇总、状态恢复和资源释放。开发时可以先运行自动回归，再用联调脚本检查单进程与分布式流程。需要观察训练速度时，使用独立的吞吐测量脚本，固定模型和输入配置进行比较。

## 开发环境与测试

从仓库根目录安装开发环境、运行测试并构建发行包：

```bash
python -m pip install build
python -m pip install -e .
MPLBACKEND=Agg python -m unittest discover -s tests -v
python -m build
```

测试会根据运行环境跳过不满足条件的用例。小数据联调脚本用于检查单进程、CPU 多进程或 CUDA 环境下的训练、全局指标和 Checkpoint 临时加载：

```bash
python tests/smoke_training.py --output /tmp/deepepochs-smoke-python
torchrun --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 tests/smoke_training.py --output /tmp/deepepochs-smoke-ddp
python tests/smoke_training.py --device cuda --precision fp16 --output /tmp/deepepochs-smoke-cuda
```

## CPU 自动回归

仓库的 [CPU 测试工作流](../.github/workflows/cpu.yml) 在提交和 Pull Request 时运行，覆盖 Python 3.10 / PyTorch 2.1 与 Python 3.12 / PyTorch 2.8。每组环境执行完整 CPU/Gloo 回归、发行包构建和严格检查；需要 CUDA 的用例按环境跳过，GPU 联调通过独立脚本执行。

## 多 GPU 联调

多 GPU 联调分别检查启动入口和训练过程：前者确认进程、设备及退出行为，后者将分布式计算与全局参考结果对照，检查训练数值和状态是否一致。

小数据脚本可以直接检查 Python 启动入口。CPU 示例显式指定两个进程，CUDA 示例自动使用可见设备，并检查实际进程数。根据可用 GPU 数量调整命令中的进程数；三卡示例采用全局批量 12，由每个进程处理 4 个样本：

```bash
python tests/smoke_launch.py --num-processes=2 --expected-world-size=2 --device=cpu --output /tmp/deepepochs-launch-cpu
python tests/smoke_launch.py --expected-world-size=4 --device=cuda --precision=fp16 --output /tmp/deepepochs-launch-cuda
CUDA_VISIBLE_DEVICES=0,1,2 python tests/smoke_launch.py --expected-world-size=3 --batch-size=12 --device=cuda --precision=fp16 --output /tmp/deepepochs-launch-three
```

训练与评估脚本检查全局指标、参数同步、梯度累积、监控器与 Checkpoint 恢复。下面以四卡 FP16 为例，先执行训练和评估，再将评估脚本生成的 Checkpoint 交给单卡脚本继续训练：

```bash
torchrun --standalone --nproc-per-node=4 tests/multigpu_training.py --expected-world-size 4 --precision fp16 --output /tmp/deepepochs-training-4-fp16.json
torchrun --standalone --nproc-per-node=4 tests/multigpu_evaluation.py --expected-world-size 4 --precision fp16 --output /tmp/deepepochs-evaluation-4-fp16
python tests/single_gpu_checkpoint.py --checkpoint /tmp/deepepochs-evaluation-4-fp16/checkpoints/multiple/checkpoint-2.ckpt --source-world-size 4 --precision fp16 --device cuda:0 --output /tmp/deepepochs-single-checkpoint
```

## 吞吐测量

吞吐以每秒处理的图像数衡量。测量脚本使用固定卷积模型和 CPU 缓存的合成图像，执行 Trainer 的完整批次训练路径，关闭 TF32。每卡 batch 固定时，全局 batch 随卡数增加，这种测量称为弱扩展。

`--warmup` 设置预热步数，`--steps` 设置计时步数，`--batch-size` 指定每卡批量。计时窗口包含 CPU 数据整理、数据传入 GPU（H2D）、模型计算、参数更新和 batch 统计。全局吞吐按全部样本数除以最慢进程的测量时间计算；每轮的指标汇总和日志位于批次计时窗口之外，完整 `fit` 耗时另行记录。

固定测量配置后，可以依次运行各精度与卡数组合，根据本机设备调整 `gpus` 列表：

```bash
for precision in fp32 fp16 bf16; do
  for gpus in 1 2 4; do
    torchrun --standalone --nproc-per-node="$gpus" tests/multigpu_benchmark.py --expected-world-size "$gpus" --precision "$precision" --warmup 20 --steps 200 --batch-size 32 --output "/tmp/deepepochs-benchmark-${gpus}-${precision}.json"
  done
done
```

测量结果写入 `--output` 指定的 JSON 文件，包含各进程的每步耗时、吞吐和峰值显存，可用于查看各进程的运行情况。

## 适用范围与约束

BF16 联调需要支持该精度的 CUDA 设备。吞吐测量采用合成 CPU 缓存数据，实际输入管线中的读取、解码和预处理会影响端到端吞吐。

---

[返回首页](../README.md) · [分布式训练](distributed.md) · [训练配置](training.md)
