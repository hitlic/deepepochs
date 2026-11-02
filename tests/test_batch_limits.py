"""批次上限只消费所需数据，并正确提交梯度累积的尾窗口。"""
import contextlib
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deepepochs import EpochTask, Trainer


class CountReads(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets
        self.reads = []

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.inputs[index] if self.targets is None else (self.inputs[index], self.targets[index])


class BatchLimitTests(unittest.TestCase):
    def trainer(self, **kwargs):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        trainer = Trainer(model, nn.functional.mse_loss, torch.optim.SGD(model.parameters(), lr=.01),
                          device='cpu', epochs=1, show_info=False, log_batch=False, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_limited_accumulation_matches_independent_sgd_and_reads_no_extra_batch(self):
        inputs = torch.arange(1., 10.).reshape(-1, 1)
        targets = torch.tensor([1., 4., 2., 9., 3., 16., 4., 25., 5.]).reshape(-1, 1)
        reference = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            reference.weight.zero_()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        for start, stop in ((0, 4), (4, 6)):
            optimizer.zero_grad()
            nn.functional.mse_loss(reference(inputs[start:stop]), targets[start:stop]).backward()
            optimizer.step()
        for source in ('fit', 'task'):
            for non_blocking in (False, True):
                with self.subTest(source=source, non_blocking=non_blocking):
                    trainer = self.trainer(gradient_accumulation_steps=2, non_blocking=non_blocking)
                    dataset = CountReads(inputs, targets)
                    loader = DataLoader(dataset, batch_size=2)
                    task = EpochTask(loader, max_batches=3 if source == 'task' else None)
                    self.quiet(trainer.fit, train_tasks=[task], limit_train_batches=3 if source == 'fit' else None)
                    torch.testing.assert_close(trainer.model.weight, reference.weight)
                    self.assertEqual(dataset.reads, list(range(6)))
                    self.assertEqual(task.batchs, 3)

    def test_fit_limits_are_per_call_and_leave_task_configuration_reusable(self):
        inputs = torch.ones(5, 1)
        train_data, val_data = [CountReads(inputs, inputs) for _ in range(2)]
        train_task = EpochTask(DataLoader(train_data, batch_size=1), name='trainset', max_batches=3)
        val_task = EpochTask(DataLoader(val_data, batch_size=1), name='valset', max_batches=3)
        train_tasks, val_tasks = [train_task], [val_task]
        trainer = self.trainer()
        self.quiet(trainer.fit, train_tasks=train_tasks, val_tasks=val_tasks,
                   limit_train_batches=1, limit_val_batches=2)
        self.assertEqual(train_data.reads, [0])
        self.assertEqual(val_data.reads, [0, 1])
        train_data.reads.clear()
        val_data.reads.clear()
        self.quiet(trainer.fit, train_tasks=train_tasks, val_tasks=val_tasks, epochs=1)
        self.assertEqual(train_data.reads, [0, 1, 2])
        self.assertEqual(val_data.reads, [0, 1, 2])
        self.assertEqual((train_task.max_batches, val_task.max_batches), (3, 3))
        self.assertEqual(train_tasks, [train_task])
        self.assertEqual(val_tasks, [val_task])
        self.assertEqual(train_task.metrics, [])
        self.assertEqual(val_task.metrics, [])

    def test_test_and_predict_limits_do_not_read_the_next_batch(self):
        trainer = self.trainer()
        inputs = torch.ones(5, 1)
        for stage in ('test', 'predict'):
            with self.subTest(stage=stage):
                dataset = CountReads(inputs, inputs if stage == 'test' else None)
                loader = DataLoader(dataset, batch_size=1)
                if stage == 'test':
                    task = EpochTask(loader, max_batches=3)
                    self.quiet(trainer.test, tasks=[task], weights='current', limit_batches=1)
                else:
                    self.assertEqual(len(list(trainer.predict(loader, limit_batches=1))), 1)
                self.assertEqual(dataset.reads, [0])
                if stage == 'test':
                    dataset.reads.clear()
                    self.quiet(trainer.test, tasks=[task], weights='current')
                    self.assertEqual(dataset.reads, [0, 1, 2])
                    self.assertEqual(task.max_batches, 3)

    def test_invalid_limits_raise_before_consuming_data(self):
        trainer = self.trainer()
        dataset = CountReads(torch.ones(2, 1), torch.ones(2, 1))
        loader = DataLoader(dataset, batch_size=1)
        for value in (0, -1, True, 1.5, '2'):
            for entry in ('task', 'train', 'val', 'test', 'predict'):
                with self.subTest(value=value, entry=entry), self.assertRaises(ValueError):
                    if entry == 'task':
                        EpochTask(loader, max_batches=value)
                    elif entry == 'train':
                        self.quiet(trainer.fit, loader, limit_train_batches=value)
                    elif entry == 'val':
                        self.quiet(trainer.fit, loader, val_dl=loader, limit_val_batches=value)
                    elif entry == 'test':
                        self.quiet(trainer.test, loader, weights='current', limit_batches=value)
                    else:
                        list(trainer.predict(loader, limit_batches=value))
        self.assertEqual(dataset.reads, [])

    def test_non_blocking_reaches_nested_inputs_labels_and_predictions(self):
        original_to = torch.Tensor.to
        for non_blocking in (False, True):
            with self.subTest(non_blocking=non_blocking):
                trainer = self.trainer(non_blocking=non_blocking)
                calls = []
                def record_to(tensor, *args, **kwargs):
                    if 'non_blocking' in kwargs:
                        calls.append((tensor, kwargs['non_blocking']))
                    return original_to(tensor, *args, **kwargs)
                inputs = (torch.ones(2, 1), {'offset': torch.ones(2, 1)})
                labels = torch.zeros(2, 1)
                with patch.object(torch.Tensor, 'to', record_to):
                    trainer.prepare_data((inputs, labels), to_device=True)
                    list(trainer.predict(DataLoader(torch.ones(1, 1), batch_size=1)))
                self.assertEqual(len(calls), 4)
                self.assertTrue(all(flag is non_blocking for _, flag in calls))
                self.assertTrue(any(tensor is labels for tensor, _ in calls))


if __name__ == '__main__':
    unittest.main()
