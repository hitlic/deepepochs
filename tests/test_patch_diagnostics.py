import tempfile
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from deepepochs.distributed import DistributedContext
from deepepochs.patches import ConfusionPatch, PatchAccumulator, ValuePatch


class BrokenAddPatch(ValuePatch):
    def add(self, other):
        raise ValueError('custom combine failed')


class BrokenForwardPatch(ValuePatch):
    def forward(self):
        raise LookupError('custom metric failed')


def _shared_error_worker(rank, store_path):
    dist.init_process_group('gloo', init_method=f'file://{store_path}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        context = DistributedContext('cpu')
        cases = [
            ('add', 'custom_sum', 'ValueError', 'in add', 'custom combine failed'),
            ('forward', 'custom_metric', 'LookupError', 'in forward', 'custom metric failed'),
            ('metadata', 'unloaded_confusion', 'AttributeError', 'in _reduction_metadata', 'confusion_matrix'),
        ]
        for phase, key, error_type, source_function, detail in cases:
            accumulator = PatchAccumulator(context)
            if rank == 1:
                if phase == 'add':
                    accumulator.add({key: BrokenAddPatch(1).load(1.)})
                    accumulator.add({key: BrokenAddPatch(1).load(2.)})
                elif phase == 'forward':
                    accumulator.add({key: BrokenForwardPatch(1).load(1.)})
                else:
                    # 忘记 load 时，元数据检查也应指出具体指标。
                    accumulator.add({key: ConfusionPatch(lambda matrix: matrix.sum())})
            try:
                accumulator.compute()
            except RuntimeError as error:
                message = str(error)
            else:
                raise AssertionError(f'{phase} failure was not shared')
            for expected in (repr(key), 'rank 1', error_type, detail,
                             'Traceback (most recent call last)', source_function):
                assert expected in message, (phase, expected, message)
            source_file = 'patches.py' if phase == 'metadata' else Path(__file__).name
            assert source_file in message, message
            received = context.all_gather_object(message)
            assert received[0] == received[1], received
        assert PatchAccumulator(context).compute() == {}
    finally:
        dist.destroy_process_group()


class PatchDiagnosticsTests(unittest.TestCase):
    def test_single_process_add_preserves_original_exception_and_frame(self):
        accumulator = PatchAccumulator(SimpleNamespace(is_distributed=False))
        accumulator.add({'custom_sum': BrokenAddPatch(1).load(1.)})
        try:
            accumulator.add({'custom_sum': BrokenAddPatch(1).load(2.)})
        except ValueError as error:
            self.assertEqual(str(error), 'custom combine failed')
            self.assertEqual(traceback.extract_tb(error.__traceback__)[-1].name, 'add')
        else:
            self.fail('custom add did not raise')

    def test_single_process_forward_preserves_original_exception_and_frame(self):
        accumulator = PatchAccumulator(SimpleNamespace(is_distributed=False))
        accumulator.add({'custom_metric': BrokenForwardPatch(1).load(1.)})
        try:
            accumulator.compute()
        except LookupError as error:
            self.assertEqual(str(error), 'custom metric failed')
            self.assertEqual(traceback.extract_tb(error.__traceback__)[-1].name, 'forward')
        else:
            self.fail('custom forward did not raise')

    def test_gloo_shares_metric_name_and_source_with_an_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_shared_error_worker, args=(str(Path(directory) / 'store'),),
                     nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
