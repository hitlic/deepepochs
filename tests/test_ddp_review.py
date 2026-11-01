"""覆盖初始化回调、任意非空指标 owner 和编译模型权重互通。"""

import os
from pathlib import Path
import socket
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

from deepepochs import Callback, EpochTask, PatchBase, Trainer
from deepepochs import metrics
from deepepochs.callbacks.check import save_state
from deepepochs.distributed import DistributedContext
from deepepochs.patches import ValuePatch, aggregate_patch_dicts


class ExactIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))
        self.register_buffer('offset', torch.zeros(2))

    def forward(self, logits):
        return logits + self.anchor * 0 + self.offset


def _trainer(model, loss, **kwargs):
    return Trainer(model, loss, device='cpu', show_info=False, log_batch=False, **kwargs)


def _initialization_check(context):
    class RandomInitialization(Callback):
        def on_before_fit(self, trainer, epochs):
            raw = context.unwrap_serialization_model(trainer.model.model)
            generator = torch.Generator().manual_seed(37 + context.rank)
            with torch.no_grad():
                for parameter in raw.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator))
                raw.offset.fill_(context.rank + 1.)
            self.initialized = {key: value.clone() for key, value in raw.state_dict().items()}

        def on_before_train_epochs(self, trainer, tasks, epoch_idx):
            raw = context.unwrap_serialization_model(trainer.model.model)
            states = context.all_gather_object(raw.state_dict())
            expected = context.broadcast_object(self.initialized if context.rank == 0 else None)
            for state in states:
                for key in expected:
                    torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
            self.checked = True

    model = nn.Linear(1, 1)
    model.register_buffer('offset', torch.tensor(0.))
    callback = RandomInitialization()
    trainer = _trainer(model, nn.functional.mse_loss, callbacks=[callback], epochs=1)
    try:
        inputs = torch.arange(4.).view(-1, 1)
        trainer.fit(DataLoader(TensorDataset(inputs, inputs * 2), batch_size=1))
        assert callback.checked
    finally:
        trainer.close()


def _metric_owner_checks(context):
    logits = torch.tensor([[4., 0.], [0., 4.], [4., 0.], [4., 0.], [4., 0.]])
    targets = torch.tensor([0, 1, 1, 1, 1])
    dataset = TensorDataset(logits, targets)
    expected_loss = nn.functional.cross_entropy(logits, targets).item()

    # 自定义采样器可以让主进程为空；TensorPatch/ConfusionPatch 的局部函数
    # 只存在于各自进程，不能通过序列化整个 Patch 来解决 owner 缺失。
    for patch_kind in ('tensor', 'confusion'):
        if patch_kind == 'tensor':
            def macro_f1(preds, labels):
                return metrics.f1(preds, labels, average='macro')
        else:
            def macro_f1(matrix):
                return metrics.f1(conf_mat=matrix, average='macro')

        loader = DataLoader(dataset, batch_size=2,
                            sampler=ExactIndicesSampler([] if context.rank == 0 else list(range(5))))
        trainer = _trainer(FixedLogits(), nn.functional.cross_entropy,
                           metrics=[macro_f1], metric_patch=patch_kind)
        try:
            result = trainer.test(loader)
            assert abs(float(result['loss']) - expected_loss) < 1e-6, result
            assert abs(float(result['macro_f1']) - 0.4) < 1e-6, result
        finally:
            trainer.close()

    def ratio(correct, count):
        return correct / count

    class RemoteAccuracyPatch(PatchBase):
        def __init__(self):
            super().__init__('remote_accuracy')
            self.metric = ratio

        def load(self, preds, labels):
            self.correct = (preds.argmax(1) == labels).sum()
            self.count = len(labels)
            return self

        def forward(self):
            return self.metric(self.correct, self.count)

        def add(self, other):
            self.correct = self.correct + other.correct
            self.count += other.count
            return self

    class DynamicTask(EpochTask):
        def evaluate_step(self, batch_x, batch_y):
            output = self.model(batch_x)
            self.loss(output, batch_y)
            if self.rank == 1:
                return {'remote': RemoteAccuracyPatch().load(output, batch_y)}
            return {}

    # 主进程这次有普通 loss，但 remote key 只由 rank 1 返回。
    indices = [0, 2] if context.rank == 0 else [1, 3, 4]
    task = DynamicTask(DataLoader(dataset, batch_size=2, sampler=ExactIndicesSampler(indices)))
    trainer = _trainer(FixedLogits(), nn.functional.cross_entropy)
    try:
        result = trainer.test(tasks=task)
        assert abs(float(result['loss']) - expected_loss) < 1e-6, result
        assert abs(float(result['remote_accuracy']) - 1 / 3) < 1e-6, result
    finally:
        trainer.close()


def _aggregation_error_checks(context):
    class LocalMetricError(ValueError):
        pass

    class CombinedFailurePatch(PatchBase):
        def load(self, value):
            self.value = value
            return self

        def forward(self):
            if self.value > 1:
                raise LocalMetricError('combined metric failed')
            return self.value

        def add(self, other):
            self.value = self.value + other.value
            return self

    try:
        aggregate_patch_dicts([{'bad': CombinedFailurePatch().load(torch.tensor(1))}], context)
    except RuntimeError as error:
        message = str(error)
        assert 'LocalMetricError' in message and 'combined metric failed' in message
    else:
        raise AssertionError('所有进程都应收到指标计算失败。')
    messages = context.all_gather_object(message)
    assert messages[0] == messages[1], messages

    # rank 1 在本地合并两个不一致的字典时失败；rank 0 不能进入后续广播等待。
    local = [{'value': ValuePatch(1).load({'a': torch.tensor(1.)})}]
    if context.rank == 1:
        local.append({'value': ValuePatch(1).load({'b': torch.tensor(2.)})})
    try:
        aggregate_patch_dicts(local, context)
    except RuntimeError as error:
        message = str(error)
        assert 'rank 1' in message and 'AssertionError' in message
    else:
        raise AssertionError('所有进程都应收到本地状态合并失败。')
    messages = context.all_gather_object(message)
    assert messages[0] == messages[1], messages


def _review_worker(rank, port, directory):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='2',
                      RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)

    def stage(name):
        (Path(directory) / f'rank-{rank}.stage').write_text(name, encoding='utf-8')

    stage('initialize-gloo')
    context = DistributedContext('cpu')
    try:
        stage('random-before-fit-callback')
        _initialization_check(context)
        stage('metric-owner-with-empty-rank-zero')
        _metric_owner_checks(context)
        stage('aggregation-error-propagation')
        _aggregation_error_checks(context)
        context.barrier()
        stage('complete')
    finally:
        context.close()


class DDPReviewTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要 Gloo')
    def test_initialization_metric_owners_and_errors(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            workers = mp.spawn(_review_worker, args=(port, directory), nprocs=2, join=False)
            deadline = time.monotonic() + 60
            try:
                while not workers.join(timeout=1):
                    if time.monotonic() >= deadline:
                        stages = {path.name: path.read_text(encoding='utf-8')
                                  for path in Path(directory).glob('rank-*.stage')}
                        self.fail(f'DDP 审阅回归在 60 秒内未完成：{stages}')
            finally:
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    def test_compiled_and_plain_checkpoints_are_interchangeable(self):
        with tempfile.TemporaryDirectory() as directory:
            for compiled_source in (False, True):
                source_raw = nn.Linear(2, 1)
                with torch.no_grad():
                    source_raw.weight.copy_(torch.tensor([[0.25, -0.5]]))
                    source_raw.bias.fill_(0.75)
                source = torch.compile(source_raw, backend='eager') if compiled_source else source_raw
                target_raw = nn.Linear(2, 1)
                target = target_raw if compiled_source else torch.compile(target_raw, backend='eager')
                source_trainer = _trainer(source, nn.functional.mse_loss)
                target_trainer = _trainer(target, nn.functional.mse_loss)
                try:
                    path = str(Path(directory) / f'compiled-{compiled_source}.ckpt')
                    save_state(source_trainer, source_trainer.model, source_trainer.opt, path, epoch=0)
                    state = torch.load(path, map_location='cpu', weights_only=False)
                    self.assertEqual(set(state['model_state']), {'weight', 'bias'})
                    target_trainer.load_checkpoint(path)
                    self.assertEqual(set(target_trainer.model.state_dict()), {'weight', 'bias'})
                    for key, value in source_raw.state_dict().items():
                        torch.testing.assert_close(target_raw.state_dict()[key], value, rtol=0, atol=0)
                    inputs = torch.tensor([[1., 2.], [2., 3.]])
                    torch.testing.assert_close(target(inputs), source(inputs), rtol=0, atol=0)
                finally:
                    source_trainer.close()
                    target_trainer.close()

if __name__ == '__main__':
    unittest.main()
