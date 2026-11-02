import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deepepochs.distributed import DistributedContext
from deepepochs.metrics import accuracy
from deepepochs.patches import (
    ConfusionMetrics, ConfusionPatch, MeanPatch, PatchAccumulator, PatchBase,
    TensorPatch, ValuePatch, aggregate_patch_dicts, run_patch_dicts,
)


def _mean(preds, targets):
    return preds.mean()


def _accuracy(matrix):
    return accuracy(conf_mat=matrix)


def _has_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_has_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_tensor(item) for item in value)
    return False


class RecordingContext(DistributedContext):
    def __init__(self):
        super().__init__('cpu')
        self.payloads = []

    def all_gather_object(self, value):
        self.payloads.append(value)
        return super().all_gather_object(value)


def _distributed_worker(rank, store_path):
    dist.init_process_group('gloo', init_method=f'file://{store_path}', rank=rank, world_size=2)
    try:
        context = RecordingContext()
        accumulator = PatchAccumulator(context)
        values = [(2, 1.), (1, 3.), (2, 5.)] if rank == 0 else [(3, 7.)]
        originals = []
        with patch('deepepochs.patches.dist.all_reduce', wraps=dist.all_reduce) as reduce:
            for size, value in values:
                predictions = torch.full((size,), value, dtype=torch.float64)
                batch = {
                    'value': ValuePatch(size).load({'weighted': value, 'constant': torch.tensor(3.)}),
                    'mean': MeanPatch(_mean, size).load(predictions, None),
                    'confusion': ConfusionPatch(_accuracy).load(
                        torch.tensor([.1, .9]), torch.tensor([rank, rank])),
                    'metrics': ConfusionMetrics(metrics='accuracy', name='C.').load(
                        torch.tensor([.1, .9]), torch.tensor([rank, rank])),
                }
                originals.append(batch)
                accumulator.add(batch)
            assert reduce.call_count == 0
            assert context.payloads == []
            actual = accumulator.compute()
            assert reduce.call_count == 7, reduce.call_count
        assert actual['value']['weighted'] == 4.5
        assert float(actual['value']['constant']) == 3.
        assert float(actual['mean']) == 4.5
        assert float(actual['confusion']) == .5
        assert float(actual['C.']['acc']) == .5
        assert len(context.payloads) == 1
        assert not _has_tensor(context.payloads[0])
        for batch, (_, value) in zip(originals, values):
            assert float(batch['value']()['weighted']) == value
            assert float(batch['mean']()) == value
            assert int(batch['confusion'].confusion_matrix.sum()) == 2

        # 空 rank 也参与归约；动态指标的本地闭包由 rank 1 计算。
        context.payloads.clear()
        accumulator = PatchAccumulator(context)
        if rank == 1:
            accumulator.add({'remote': ValuePatch(3).load(torch.tensor(7.)),
                             'tensor': TensorPatch(lambda preds, _: preds.median()).load(
                                 torch.tensor([4., 9., 1.]), None)})
        actual = accumulator.compute()
        assert float(actual['remote']) == 7.
        assert float(actual['tensor']) == 4.
        assert not _has_tensor(context.payloads[0])
        assert set(context.payloads[1]) == ({'tensor'} if rank == 1 else set())

        # 局部类、覆盖内置 add / forward 的子类和完整预测都使用通用累计。
        class HitsPatch(PatchBase):
            def load(self, value):
                self.value = value
                return self

            def add(self, other):
                self.value += other.value
                return self

            def forward(self):
                return self.value

        class AdjustedValuePatch(ValuePatch):
            def add(self, other):
                super().add(other)
                self.batch_value += 1
                return self

            def forward(self):
                return super().forward() * 2

        context.payloads.clear()
        accumulator = PatchAccumulator(context)
        median = lambda preds, _: preds.median()
        for value in ([1., 10.] if rank == 0 else [5.]):
            accumulator.add({
                'exact': TensorPatch(median).load(torch.tensor([value]), None),
                'custom': HitsPatch(name='hits').load(1),
                'subclass': AdjustedValuePatch(1).load(value),
            })
        with patch('deepepochs.patches.dist.all_reduce', wraps=dist.all_reduce) as reduce:
            actual = accumulator.compute()
            assert reduce.call_count == 0
        assert float(actual['exact']) == 5.
        assert actual['hits'] == 3
        assert actual['subclass'] == 12.  # (1 + 10 + 5 + 两次 add 的偏移) / 3 * 2
        assert all(value is None for value in context.payloads[0].values())
        assert _has_tensor(context.payloads[1])

        # 跨 rank 的 dtype 不同沿用 Patch 相加时的类型提升。
        context.payloads.clear()
        dtype = torch.float32 if rank == 0 else torch.float64
        accumulator = PatchAccumulator(context).add({'mixed': ValuePatch(1).load(torch.tensor(rank + 1., dtype=dtype))})
        with patch('deepepochs.patches.dist.all_reduce', wraps=dist.all_reduce) as reduce:
            actual = accumulator.compute()
            assert reduce.call_count == 0
        assert actual['mixed'].dtype == torch.float64
        assert float(actual['mixed']) == 1.5

        # 本地累计失败先汇合，再让所有进程收到同一个错误。
        accumulator = PatchAccumulator(context)
        if rank == 1:
            accumulator.add({'bad': ValuePatch(1).load({'a': 1.})})
            accumulator.add({'bad': ValuePatch(1).load({'b': 2.})})
        try:
            accumulator.compute()
        except RuntimeError as error:
            assert 'rank 1' in str(error) and 'AssertionError' in str(error)
        else:
            raise AssertionError('local accumulation failure was not shared')

        class LocalMetricError(ValueError):
            pass

        def bad_metric(matrix):
            raise LocalMetricError('metric failed')

        accumulator = PatchAccumulator(context)
        if rank == 1:
            accumulator.add({'bad': ConfusionPatch(bad_metric).load(
                torch.tensor([.1, .9]), torch.tensor([0, 1]))})
        try:
            accumulator.compute()
        except RuntimeError as error:
            assert 'rank 1' in str(error) and 'LocalMetricError' in str(error)
        else:
            raise AssertionError('owner metric failure was not shared')

        # 不相容的矩阵形状由原有 add 语义明确报错。
        accumulator = PatchAccumulator(context).add({'bad': ConfusionPatch(_accuracy).load(
            torch.ones(1, rank + 2), torch.tensor([0]))})
        try:
            accumulator.compute()
        except RuntimeError as error:
            assert 'AssertionError' in str(error)
        else:
            raise AssertionError('incompatible confusion matrices were accepted')
        assert PatchAccumulator(context).compute() == {}
    finally:
        dist.destroy_process_group()


class PatchStreamingTests(unittest.TestCase):
    def test_long_stream_retains_one_state_per_builtin_metric(self):
        accumulator = PatchAccumulator(SimpleNamespace(is_distributed=False))
        refs = []
        for index in range(512):
            batch = {
                'value': ValuePatch(2).load(float(index)),
                'mean': MeanPatch(_mean, 2).load(torch.tensor([float(index)] * 2), None),
                'confusion': ConfusionPatch(_accuracy).load(torch.tensor([.1, .9]), torch.tensor([0, 1])),
                'metrics': ConfusionMetrics(metrics='accuracy').load(torch.tensor([.1, .9]), torch.tensor([0, 1])),
            }
            refs.extend(weakref.ref(value) for value in batch.values())
            accumulator.add(batch)
            self.assertEqual(len(accumulator.patches), 4)
        del batch
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(accumulator.patches['mean'].batch_value.numel(), 1)
        self.assertEqual(accumulator.patches['confusion'].confusion_matrix.numel(), 4)
        actual = accumulator.compute()
        self.assertEqual(actual['value'], 255.5)
        self.assertEqual(float(actual['mean']), 255.5)
        self.assertEqual(float(actual['confusion']), 1.)

    def test_weighted_statistics_and_original_batches_match_sum(self):
        batches = []
        for size, value in [(2, 1.), (1, 7.), (3, 2.)]:
            inputs = torch.full((size,), value, dtype=torch.float64, requires_grad=True)
            batches.append({
                'value': ValuePatch(torch.tensor(size)).load({'x': inputs.mean(), 'y': value}),
                'mean': MeanPatch(lambda preds, _: {'x': preds.mean(), 'y': value}, size).load(inputs, None),
                'confusion': ConfusionPatch(_accuracy).load(torch.tensor([.1, .9]), torch.tensor([0, 1])),
                'metrics': ConfusionMetrics().load(torch.tensor([.1, .9]), torch.tensor([0, 1])),
            })
        expected = run_patch_dicts(batches)
        accumulator = PatchAccumulator(SimpleNamespace(is_distributed=False))
        for batch in batches:
            accumulator.add(batch)
        actual = accumulator.compute()
        for key in actual:
            if isinstance(actual[key], dict):
                for field in actual[key]:
                    torch.testing.assert_close(actual[key][field], expected[key][field])
            else:
                torch.testing.assert_close(actual[key], expected[key])
        for batch, (size, value) in zip(batches, [(2, 1.), (1, 7.), (3, 2.)]):
            self.assertEqual(int(batch['value'].batch_size), size)
            self.assertEqual(float(batch['value']()['x']), value)
            self.assertFalse(batch['value'].batch_value['x'].requires_grad)
            self.assertEqual(int(batch['confusion'].confusion_matrix.sum()), 2)

    def test_tensor_fallback_keeps_full_predictions(self):
        accumulator = PatchAccumulator(SimpleNamespace(is_distributed=False))
        metric = lambda preds, _: preds.median()
        originals = []
        for values in ([1., 9.], [5.], [3., 8.]):
            batch = TensorPatch(metric).load(torch.tensor(values), None)
            originals.append(batch)
            accumulator.add({'median': batch})
        self.assertEqual(len(accumulator.patches['median'].batch_preds), 3)
        self.assertEqual(float(accumulator.compute()['median']), 5.)
        self.assertEqual([len(batch.batch_preds) for batch in originals], [1, 1, 1])
        self.assertEqual(aggregate_patch_dicts([], SimpleNamespace(is_distributed=False)), {})

    def test_gloo_reduction_fallback_empty_rank_and_shared_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_worker, args=(str(Path(directory) / 'store'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
