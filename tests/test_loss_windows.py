"""梯度累积的损失窗口、独立调度与中断重试。"""
import contextlib
import io
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, EpochTask, Optimizer, Trainer
from deepepochs.loops import StopLoopException


class RecordLoss:
    def __init__(self): self.values = []
    def step(self, loss): self.values.append(float(loss))


class SeparateLosses(EpochTask):
    def train_step(self, batch_x, batch_y):
        self.loss(self.model(batch_x)[:, :1], batch_y[:, :1], do_optimize=False, do_metric=False)
        self.optimize(0)
        self.loss(self.model(batch_x)[:, 1:], batch_y[:, 1:], do_optimize=False, do_metric=False)
        self.optimize(1)


class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(1, 1, bias=False)
        self.right = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.left.weight.zero_()
            self.right.weight.zero_()
    def forward(self, x):
        return torch.cat((self.left(x), self.right(x)), dim=1)


class InterruptOnce(Callback):
    def __init__(self, error_type=StopLoopException, phase='after_batch', batch_idx=0):
        super().__init__()
        self.error_type = error_type
        self.phase = phase
        self.batch_idx = batch_idx
        self.enabled = True

    def on_after_backward(self, trainer, loss):
        if self.enabled and self.phase == 'after_backward':
            self.enabled = False
            raise self.error_type('interrupt accumulation')

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        if self.enabled and self.phase == 'after_batch' and batch_idx == self.batch_idx:
            self.enabled = False
            raise self.error_type('interrupt accumulation')


class ManualSubBatches(EpochTask):
    def train_step(self, batch_x, batch_y):
        for inputs, targets in zip(batch_x.split(1), batch_y.split(1)):
            self.loss(self.model(inputs), targets, loss_adjust=len(inputs) / len(batch_x),
                      do_optimize=False, do_metric=False)
        self.optimize()


class LossWindowTests(unittest.TestCase):
    def test_each_optimizer_receives_its_own_accumulated_mean(self):
        model = TwoHeads()
        schedulers = [RecordLoss(), RecordLoss()]
        # lr=0 固定预测，独立参考损失分别是 (1+9)/2、(100+900)/2。
        optimizers = [Optimizer(torch.optim.SGD((model.left if i == 0 else model.right).parameters(), lr=0.), schedulers[i],
                                sched_on='step', sched_with_loss=True) for i in range(2)]
        trainer = Trainer(model, nn.functional.mse_loss, optimizers, epochs=1,
                          gradient_accumulation_steps=2, device='cpu', show_info=False, log_batch=False)
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.tensor([[1., 10.], [3., 30.]])), batch_size=1)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(train_tasks=[SeparateLosses(loader)])
        self.assertEqual(schedulers[0].values, [5.])
        self.assertEqual(schedulers[1].values, [500.])


    def trainer(self, callback, accumulation_steps=2, momentum=0.):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        scheduler = RecordLoss()
        optimizer = Optimizer(torch.optim.SGD(model.parameters(), lr=.1, momentum=momentum), scheduler,
                              sched_on='step', sched_with_loss=True)
        trainer = Trainer(model, nn.functional.mse_loss, optimizer, epochs=1,
                          gradient_accumulation_steps=accumulation_steps, callbacks=[callback],
                          device='cpu', show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        return trainer, model, scheduler

    def loader(self, targets, batch_size=1):
        targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        return DataLoader(TensorDataset(torch.ones_like(targets), targets), batch_size=batch_size)

    def test_interrupted_accumulation_is_discarded_before_retry(self):
        for phase in ('after_backward', 'after_batch'):
            for error_type in (StopLoopException, RuntimeError, KeyboardInterrupt):
                with self.subTest(phase=phase, error=error_type.__name__):
                    stop = InterruptOnce(error_type, phase)
                    trainer, model, scheduler = self.trainer(stop)
                    loader = self.loader([1., 3.])
                    with contextlib.redirect_stdout(io.StringIO()):
                        if error_type is RuntimeError:
                            with self.assertRaisesRegex(RuntimeError, 'interrupt accumulation'):
                                trainer.fit(loader)
                        else:
                            trainer.fit(loader)
                    self.assertEqual(float(model.weight.detach()), 0.)
                    self.assertEqual(scheduler.values, [])
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit(loader)
                    # 两批的独立参考：loss=(1+9)/2=5，梯度=-4，SGD更新为0.4。
                    self.assertAlmostEqual(float(model.weight.detach()), .4, places=6)
                    self.assertEqual(scheduler.values, [5.])

    def test_interruption_keeps_committed_parameter_and_momentum_updates(self):
        stop = InterruptOnce(RuntimeError, batch_idx=2)
        trainer, model, scheduler = self.trainer(stop, momentum=.9)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'interrupt accumulation'):
                trainer.fit(self.loader([1., 3., 10., 20.]))
        self.assertAlmostEqual(float(model.weight.detach()), .4, places=6)
        self.assertEqual(scheduler.values, [5.])
        self.assertIsNone(model.weight.grad)

        reference = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            reference.weight.zero_()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.1, momentum=.9)
        inputs, targets = next(iter(self.loader([1., 3.], batch_size=2)))
        expected_losses = []
        for _ in range(2):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(reference(inputs), targets)
            expected_losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(self.loader([1., 3.]))
        torch.testing.assert_close(model.weight, reference.weight)
        for actual, expected in zip(scheduler.values, expected_losses):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_interrupted_optimizers_restart_with_independent_loss_windows(self):
        model = TwoHeads()
        schedulers = [RecordLoss(), RecordLoss()]
        optimizers = [Optimizer(torch.optim.SGD(head.parameters(), lr=.1), scheduler,
                                sched_on='step', sched_with_loss=True)
                      for head, scheduler in zip((model.left, model.right), schedulers)]
        trainer = Trainer(model, nn.functional.mse_loss, optimizers, epochs=1,
                          gradient_accumulation_steps=2, callbacks=[InterruptOnce()],
                          device='cpu', show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.tensor([[1., 10.], [3., 30.]])),
                            batch_size=1)
        task = SeparateLosses(loader)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(train_tasks=[task])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(train_tasks=[task])
        self.assertEqual([scheduler.values for scheduler in schedulers], [[5.], [500.]])
        self.assertAlmostEqual(float(model.left.weight.detach()), .4, places=6)
        self.assertAlmostEqual(float(model.right.weight.detach()), 4., places=6)

    def test_interrupted_manual_sub_batches_do_not_leak_into_retry(self):
        trainer, model, scheduler = self.trainer(InterruptOnce(phase='after_backward'), accumulation_steps=1)
        task = ManualSubBatches(self.loader([1., 3.], batch_size=2))
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(train_tasks=[task])
        self.assertIsNone(model.weight.grad)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(train_tasks=[task])
        self.assertAlmostEqual(float(model.weight.detach()), .4, places=6)
        self.assertEqual(scheduler.values, [5.])


if __name__ == '__main__': unittest.main()
