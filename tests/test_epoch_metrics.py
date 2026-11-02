"""完整数据指标在任务结束求值，逐批损失继续用于回调。"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

from deepepochs import Callback, EpochTask, LRFindCallback, TensorPatch, Trainer


def strict_auc(predictions, targets):
    positives = predictions[:, 1][targets == 1]
    negatives = predictions[:, 1][targets == 0]
    if not positives.numel() or not negatives.numel():
        raise ValueError('AUROC requires both positive and negative targets')
    differences = positives[:, None] - negatives
    return ((differences > 0).float() + 0.5 * (differences == 0).float()).mean()


def model():
    result = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        result.weight.copy_(torch.tensor([[0.], [1.]]))
    return result


def dataset():
    return TensorDataset(torch.arange(4.).view(-1, 1), torch.tensor([0, 0, 1, 1]))


class RecordBatches(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = {'train': [], 'val': [], 'test': []}

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.metrics['train'].append(dict(metrics))

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.metrics['val'].append(dict(metrics))

    def on_after_test_batch(self, trainer, metrics, batch_idx):
        self.metrics['test'].append(dict(metrics))


class RankIndices(Sampler):
    def __init__(self, rank):
        self.indices = [0, 1] if rank == 0 else [2, 3]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


def _gloo_auc_worker(rank, store_path):
    dist.init_process_group('gloo', init_method=f'file://{store_path}', rank=rank, world_size=2)
    try:
        recorder = RecordBatches()
        loader = DataLoader(dataset(), batch_size=1, sampler=RankIndices(rank))
        with Trainer(model(), nn.functional.cross_entropy, device='cpu',
                     metrics=[strict_auc], callbacks=[recorder], compute_batch_metrics=False,
                     log_batch=False, show_info=False) as trainer:
            with contextlib.redirect_stdout(io.StringIO()):
                result = trainer.test(tasks=[EpochTask(loader, name='global')])
            assert float(result['global/strict_auc']) == 1., result
            assert all(set(values) == {'global/loss'} for values in recorder.metrics['test'])
            inputs, targets = dataset().tensors
            expected = nn.functional.cross_entropy(model()(inputs), targets).item()
            assert abs(float(result['global/loss']) - expected) < 1e-6
    finally:
        dist.destroy_process_group()


class EpochMetricTests(unittest.TestCase):
    def trainer(self, **kwargs):
        net = model()
        trainer = Trainer(net, nn.functional.cross_entropy,
                          torch.optim.SGD(net.parameters(), lr=0.), device='cpu',
                          log_batch=False, show_info=False, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_full_dataset_auc_runs_once_per_task_with_single_class_batches(self):
        sizes = []

        def auc(predictions, targets):
            sizes.append(len(targets))
            return strict_auc(predictions, targets)

        recorder = RecordBatches()
        trainer = self.trainer(metrics=[auc], callbacks=[recorder], compute_batch_metrics=False)
        task = EpochTask(DataLoader(dataset(), batch_size=2), name='main')
        history = self.quiet(trainer.fit, train_tasks=[task], val_tasks=[task], epochs=1)
        result = self.quiet(trainer.test, tasks=[task])
        self.assertEqual(sizes, [4, 4, 4])
        self.assertEqual(float(history['train']['main/auc'][0]), 1.)
        self.assertEqual(float(history['val']['main/auc'][0]), 1.)
        self.assertEqual(float(result['main/auc']), 1.)
        for batches in recorder.metrics.values():
            self.assertEqual(len(batches), 2)
            self.assertTrue(all(set(values) == {'main/loss'} for values in batches))

    def test_default_still_supplies_batch_metrics_to_callbacks(self):
        sizes = []

        def average(predictions, targets):
            sizes.append(len(targets))
            return predictions[:, 1].mean()

        recorder = RecordBatches()
        trainer = self.trainer(metrics=[average], callbacks=[recorder])
        result = self.quiet(trainer.test, DataLoader(dataset(), batch_size=2))
        self.assertEqual(sizes, [2, 2, 4])
        self.assertEqual(set(result), {'loss', 'average'})
        self.assertTrue(all(set(values) == {'loss', 'average'} for values in recorder.metrics['test']))

    def test_evaluation_without_loss_has_empty_batch_metrics_and_full_auc(self):
        recorder = RecordBatches()
        trainer = self.trainer(compute_batch_metrics=False, callbacks=[recorder])
        task = EpochTask(DataLoader(dataset(), batch_size=2), name='evaluation',
                         do_loss=False, metrics=[strict_auc])
        result = self.quiet(trainer.test, tasks=[task])
        self.assertEqual(set(result), {'evaluation/strict_auc'})
        self.assertEqual(float(result['evaluation/strict_auc']), 1.)
        self.assertEqual(recorder.metrics['test'], [{}, {}])

    def test_custom_tensor_patch_is_evaluated_on_the_full_dataset(self):
        class AucTask(EpochTask):
            def evaluate_step(self, batch_x, batch_y):
                predictions = self.model(batch_x)
                self.loss(predictions, batch_y)
                return {'roc': TensorPatch(strict_auc).load(predictions, batch_y)}

        trainer = self.trainer(compute_batch_metrics=False)
        task = AucTask(DataLoader(dataset(), batch_size=2), name='custom')
        result = self.quiet(trainer.test, tasks=[task])
        self.assertEqual(set(result), {'custom/loss', 'custom/roc'})
        self.assertEqual(float(result['custom/roc']), 1.)

    def test_named_lr_finder_receives_batch_loss_when_auc_is_deferred(self):
        finder = LRFindCallback(max_batch=4, min_lr=1e-4, max_lr=1e-3)
        trainer = self.trainer(compute_batch_metrics=False, metrics=[strict_auc], callbacks=[finder])
        # 两类批次的损失量级接近，使学习率查找按max_batch结束。
        samples = TensorDataset(torch.tensor([[-.2], [-.1], [.1], [.2]]), torch.tensor([0, 0, 1, 1]))
        task = EpochTask(DataLoader(samples, batch_size=2), name='main')
        with patch('deepepochs.callbacks.lr_find.plt.show'):
            history = self.quiet(trainer.fit, train_tasks=[task], epochs=3)
        plt.close('all')
        self.assertEqual(finder.batches_seen, 4)
        self.assertEqual(len(finder.losses), 4)
        self.assertTrue(all(torch.isfinite(torch.tensor(finder.losses))))
        self.assertEqual(float(history['train']['main/strict_auc'][0]), 1.)

    def test_gloo_aggregates_auc_when_each_rank_has_only_one_class(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_gloo_auc_worker, args=(str(Path(directory) / 'store'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
