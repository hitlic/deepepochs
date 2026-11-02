"""多任务指标从回调、历史到调度器保持完整名称。"""
import contextlib
import io
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from deepepochs import Callback, EpochTask, LogCallback, Optimizer, Trainer


class RecordMetrics(Callback):
    def __init__(self):
        super().__init__()
        self.events = []

    def record(self, stage, phase, metrics):
        self.events.append((stage, phase, {key: float(value) for key, value in metrics.items()}))

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.record('train', 'batch', metrics)

    def on_after_train_epoch(self, trainer, task, metrics):
        self.record('train', 'epoch', metrics)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.record('val', 'batch', metrics)

    def on_after_val_epoch(self, trainer, task, metrics):
        self.record('val', 'epoch', metrics)

    def on_after_test_batch(self, trainer, metrics, batch_idx):
        self.record('test', 'batch', metrics)

    def on_after_test_epoch(self, trainer, task, metrics):
        self.record('test', 'epoch', metrics)


class RecordLoss:
    def __init__(self):
        self.values = []

    def step(self, loss):
        self.values.append(float(loss))


class NamedTaskTests(unittest.TestCase):
    def loader(self, target):
        return DataLoader(TensorDataset(torch.ones(2, 1), torch.full((2, 1), float(target))), batch_size=1)

    def trainer(self, callbacks=None, optimizers=None, epochs=1):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        opt = torch.optim.SGD(model.parameters(), lr=0.) if optimizers is None else optimizers(model)
        trainer = Trainer(model, nn.functional.mse_loss, opt, callbacks=callbacks,
                          device='cpu', epochs=epochs, show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        return trainer

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_callback_keys_and_fit_test_results_keep_task_names(self):
        for names in ((None,), ('source', 'domain')):
            with self.subTest(names=names):
                recorder = RecordMetrics()
                trainer = self.trainer([recorder])
                tasks = [EpochTask(self.loader(2 * index + 1), name=name) for index, name in enumerate(names)]
                expected = {('loss' if name is None else f'{name}/loss'): float((2 * index + 1) ** 2)
                            for index, name in enumerate(names)}
                history = self.quiet(trainer.fit, train_tasks=tasks, val_tasks=tasks)
                result = self.quiet(trainer.test, tasks=tasks, weights='current')
                self.assertEqual({key: float(value) for key, value in result.items()}, expected)
                for stage in ('train', 'val'):
                    self.assertEqual(set(history[stage]), set(expected))
                    for key, value in expected.items():
                        self.assertEqual(list(history[stage][key]), [value])
                for stage in ('train', 'val', 'test'):
                    for phase, repeats in (('batch', 2), ('epoch', 1)):
                        actual = [metrics for event_stage, event_phase, metrics in recorder.events
                                  if (event_stage, event_phase) == (stage, phase)]
                        self.assertEqual(actual, [{key: value} for key, value in expected.items() for _ in range(repeats)])

    def test_duplicate_names_and_unnamed_losses_raise(self):
        for name in (None, 'domain'):
            for operation in ('fit', 'test'):
                with self.subTest(name=name, operation=operation):
                    trainer = self.trainer()
                    tasks = [EpochTask(self.loader(target), name=name) for target in (1, 3)]
                    with self.assertRaisesRegex(ValueError, 'loss'):
                        if operation == 'fit':
                            self.quiet(trainer.fit, train_tasks=tasks)
                        else:
                            self.quiet(trainer.test, tasks=tasks, weights='current')
        for name in ('', 3, False):
            with self.subTest(invalid_name=name), self.assertRaises(ValueError):
                EpochTask(self.loader(1), name=name)

    def test_each_optimizer_monitors_its_own_task_and_stage(self):
        for stage in ('train', 'val'):
            with self.subTest(stage=stage):
                schedulers = [RecordLoss(), RecordLoss()]
                def optimizers(model):
                    return [Optimizer(torch.optim.SGD(model.parameters(), lr=0.), scheduler,
                                      sched_with_loss=True, monitor=f'{name}/loss', on_stage=stage)
                            for name, scheduler in zip(('source', 'domain'), schedulers)]
                trainer = self.trainer(optimizers=optimizers, epochs=3)
                train_tasks = [EpochTask(self.loader(target), name=name)
                               for name, target in (('source', 1), ('domain', 3))]
                val_tasks = [EpochTask(self.loader(target), name=name)
                             for name, target in (('source', 2), ('domain', 4))]
                self.quiet(trainer.fit, train_tasks=train_tasks, val_tasks=val_tasks, val_freq=2)
                expected = [[1., 1., 1.], [9., 9., 9.]] if stage == 'train' else [[4.], [16.]]
                self.assertEqual([scheduler.values for scheduler in schedulers], expected)

    def test_missing_validation_monitor_errors_only_when_validation_runs(self):
        scheduler = RecordLoss()
        trainer = self.trainer(optimizers=lambda model: Optimizer(
            torch.optim.SGD(model.parameters(), lr=0.), scheduler, sched_with_loss=True,
            monitor='missing/loss', on_stage='val'), epochs=1)
        self.quiet(trainer.fit, self.loader(1), val_dl=self.loader(3), val_freq=2)
        self.assertEqual(scheduler.values, [])
        with self.assertRaisesRegex(ValueError, 'missing/loss'):
            self.quiet(trainer.fit, self.loader(1), val_dl=self.loader(3), val_freq=2, epochs=1)
        self.assertEqual(scheduler.values, [])

    def test_default_scheduler_prefers_unique_validation_loss_then_training_loss(self):
        for name in (None, 'domain'):
            with self.subTest(name=name):
                scheduler = RecordLoss()
                trainer = self.trainer(optimizers=lambda model: Optimizer(
                    torch.optim.SGD(model.parameters(), lr=0.), scheduler, sched_with_loss=True), epochs=3)
                self.quiet(trainer.fit, train_tasks=[EpochTask(self.loader(1), name=name)],
                           val_tasks=[EpochTask(self.loader(3), name=name)], val_freq=2)
                self.assertEqual(scheduler.values, [1., 9., 1.])

    def test_ambiguous_default_scheduler_requires_monitor(self):
        scheduler = RecordLoss()
        trainer = self.trainer(optimizers=lambda model: Optimizer(
            torch.optim.SGD(model.parameters(), lr=0.), scheduler, sched_with_loss=True))
        with self.assertRaisesRegex(ValueError, 'monitor'):
            self.quiet(trainer.fit, train_tasks=[EpochTask(self.loader(1), name='source'),
                                               EpochTask(self.loader(3), name='domain')])
        self.assertEqual(scheduler.values, [])

    def test_log_aligns_named_tasks_on_the_same_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = LogCallback(directory)
            trainer = self.trainer([callback], epochs=3)
            tasks = [EpochTask(self.loader(target), name=name) for name, target in (('source', 1), ('domain', 3))]
            self.quiet(trainer.fit, train_tasks=tasks, val_tasks=tasks, val_freq=2)
            trainer.logger.flush()
            events = EventAccumulator(callback.logger.log_dir).Reload()
            for name in ('source', 'domain'):
                for stage, expected in (('train', [1, 2, 3]), ('val', [2])):
                    with self.subTest(name=name, stage=stage):
                        self.assertEqual([event.step for event in events.Scalars(f'{name}/loss/epoch/{stage}')], expected)


if __name__ == '__main__':
    unittest.main()
