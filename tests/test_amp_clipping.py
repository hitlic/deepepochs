"""非有限优化器的梯度不应污染另一优化器的裁剪和更新。 """
import contextlib
import io
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, EpochTask, GradClipCallback, Optimizer, Trainer


class TwoParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.bad = nn.Parameter(torch.tensor(1.))
        self.good = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.bad * x, self.good * x


class InterruptAmpOnce(Callback):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.enabled = True

    def on_after_backward(self, trainer, loss):
        if self.enabled and self.phase == 'after_backward':
            self.enabled = False
            raise RuntimeError('interrupt AMP accumulation')

    def on_before_optimize(self, trainer, optimize_index):
        if self.enabled and self.phase == 'before_optimize':
            self.enabled = False
            raise RuntimeError('interrupt AMP accumulation')

    def on_after_optimize(self, trainer, optimize_index):
        if self.enabled and self.phase == 'after_first_optimizer' and optimize_index == 0:
            self.enabled = False
            raise RuntimeError('interrupt AMP accumulation')


class RecordStepLoss:
    def __init__(self):
        self.values = []

    def step(self, loss):
        self.values.append(float(loss))


def check_nonfinite_clipping(test, device, scaler):
    model = TwoParameters().to(device)
    # 参数组分别决定溢出；正常组的更新应等于单独裁剪该组的参考值。
    raw = [torch.optim.SGD([parameter], lr=.1) for parameter in (model.bad, model.good)]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.5) for opt in raw]
    opts = [Optimizer(opt, scheduler, sched_on='step') for opt, scheduler in zip(raw, schedulers)]
    def loss(outputs, targets):
        bad, good = outputs
        return (bad * float('nan')).mean() + good.square().mean()
    trainer = Trainer(model, loss, opts, device=device, epochs=1,
                      callbacks=[GradClipCallback(1., norm_type=2)], show_info=False, log_batch=False)
    trainer.scaler = scaler
    loader = DataLoader(TensorDataset(torch.ones(1), torch.zeros(1)), batch_size=1)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.fit(loader)
    test.assertEqual(float(model.bad.detach()), 1.)
    test.assertAlmostEqual(float(model.good.detach()), .9, places=5)
    test.assertEqual(schedulers[0].last_epoch, 0)
    test.assertEqual(schedulers[1].last_epoch, 1)
    trainer.close()


class CpuAmpClippingTests(unittest.TestCase):
    def test_finite_optimizers_keep_the_joint_norm(self):
        model = TwoParameters()
        optimizers = [torch.optim.SGD([p], lr=.1) for p in (model.bad, model.good)]
        trainer = Trainer(model, lambda outputs, targets: sum(value.square().mean() for value in outputs),
                          optimizers, device='cpu', epochs=1,
                          callbacks=[GradClipCallback(1., norm_type=2)], show_info=False, log_batch=False)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(DataLoader(TensorDataset(torch.ones(1), torch.zeros(1)), batch_size=1))
        expected = 1 - .1 / math.sqrt(2)
        self.assertAlmostEqual(float(model.bad.detach()), expected, places=6)
        self.assertAlmostEqual(float(model.good.detach()), expected, places=6)
        trainer.close()

    @unittest.skipUnless(hasattr(torch.amp, 'GradScaler'), 'CPU GradScaler需要较新的PyTorch')
    def test_nonfinite_optimizer_does_not_corrupt_finite_optimizer(self):
        with patch('deepepochs.callbacks.grad_clip.torch.isfinite',
                   side_effect=AssertionError('AMP应复用GradScaler的溢出记录')):
            check_nonfinite_clipping(self, 'cpu', torch.amp.GradScaler('cpu', init_scale=128.))

    @unittest.skipUnless(hasattr(torch.amp, 'GradScaler'), 'CPU GradScaler需要较新的PyTorch')
    def test_amp_joint_norm_deduplicates_shared_parameters_and_skips_empty_group(self):
        model = TwoParameters()
        unused = nn.Parameter(torch.tensor(1.))
        optimizers = [torch.optim.SGD(model.parameters(), lr=.1),
                      torch.optim.SGD([model.bad], lr=.1),
                      torch.optim.SGD([unused], lr=.1)]
        trainer = Trainer(model, lambda outputs, targets: sum(value.square().mean() for value in outputs),
                          optimizers, device='cpu', epochs=1,
                          callbacks=[GradClipCallback(1., norm_type=2)], show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        trainer.scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        with patch('deepepochs.callbacks.grad_clip.torch.isfinite',
                   side_effect=AssertionError('AMP应复用GradScaler的溢出记录')):
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.fit(DataLoader(TensorDataset(torch.ones(1), torch.zeros(1)), batch_size=1))
        # 两个有梯度的参数联合裁剪，共享参数在norm中只算一次，但分别参与两次更新。
        self.assertAlmostEqual(float(model.bad.detach()), 1 - .2 / math.sqrt(2), places=6)
        self.assertAlmostEqual(float(model.good.detach()), 1 - .1 / math.sqrt(2), places=6)
        self.assertEqual(float(unused.detach()), 1.)
        self.assertEqual(trainer.scaler.get_scale(), 128.)

    def test_fp32_nonfinite_group_does_not_pollute_finite_group(self):
        model = TwoParameters()
        optimizers = [torch.optim.SGD([p], lr=.1) for p in (model.bad, model.good)]
        trainer = Trainer(model, lambda outputs, targets: (outputs[0] * float('nan')).mean()
                          + outputs[1].square().mean(), optimizers, device='cpu', epochs=1,
                          callbacks=[GradClipCallback(1., norm_type=2)], show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit(DataLoader(TensorDataset(torch.ones(1), torch.zeros(1)), batch_size=1))
        # FP32不跳过坏组的更新，但联合裁剪仍需保护正常组。
        self.assertTrue(math.isnan(float(model.bad.detach())))
        self.assertAlmostEqual(float(model.good.detach()), .9, places=5)


    @unittest.skipUnless(hasattr(torch.amp, 'GradScaler'), 'CPU GradScaler需要较新的PyTorch')
    def test_interrupted_amp_window_restarts_without_stale_scaler_state(self):
        for phase in ('after_backward', 'before_optimize'):
            with self.subTest(phase=phase):
                model = nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.zero_()
                scheduler = RecordStepLoss()
                optimizer = Optimizer(torch.optim.SGD(model.parameters(), lr=.1), scheduler,
                                      sched_on='step', sched_with_loss=True)
                trainer = Trainer(model, nn.functional.mse_loss, optimizer, device='cpu', epochs=1,
                                  gradient_accumulation_steps=2, callbacks=[InterruptAmpOnce(phase)],
                                  show_info=False, log_batch=False)
                self.addCleanup(trainer.close)
                trainer.scaler = torch.amp.GradScaler('cpu', init_scale=128., growth_interval=100)
                initial_scale = trainer.scaler.state_dict()
                loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.tensor([[1.], [3.]])), batch_size=1)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'interrupt AMP accumulation'):
                        trainer.fit(loader)
                self.assertEqual(float(model.weight.detach()), 0.)
                self.assertIsNone(model.weight.grad)
                self.assertEqual(scheduler.values, [])
                self.assertEqual(trainer.scaler.state_dict(), initial_scale)
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.fit(loader)
                self.assertAlmostEqual(float(model.weight.detach()), .4, places=6)
                self.assertEqual(scheduler.values, [5.])
                self.assertEqual(trainer.scaler.get_scale(), 128.)
                self.assertEqual(trainer.scaler.state_dict()['_growth_tracker'], 1)

    @unittest.skipUnless(hasattr(torch.amp, 'GradScaler'), 'CPU GradScaler需要较新的PyTorch')
    def test_partial_amp_update_keeps_committed_optimizer_and_discards_other_window(self):
        class SeparateParameters(EpochTask):
            def train_step(self, batch_x, batch_y):
                self.loss(self.model(batch_x)[0], batch_y, do_optimize=False, do_metric=False)
                self.optimize(0)
                self.loss(self.model(batch_x)[1], batch_y, do_optimize=False, do_metric=False)
                self.optimize(1)

        class FailOnceLoss(RecordStepLoss):
            def __init__(self):
                super().__init__()
                self.enabled = True

            def step(self, loss):
                super().step(loss)
                if self.enabled:
                    self.enabled = False
                    raise RuntimeError('interrupt AMP accumulation')

        for interruption in ('callback', 'scheduler'):
            with self.subTest(interruption=interruption):
                model = TwoParameters()
                schedulers = [FailOnceLoss() if interruption == 'scheduler' else RecordStepLoss(), RecordStepLoss()]
                optimizers = [Optimizer(torch.optim.SGD([parameter], lr=.1), scheduler,
                                        sched_on='step', sched_with_loss=True)
                              for parameter, scheduler in zip((model.bad, model.good), schedulers)]
                trainer = Trainer(model, nn.functional.mse_loss, optimizers, device='cpu', epochs=1,
                                  gradient_accumulation_steps=2,
                                  callbacks=[InterruptAmpOnce('after_first_optimizer' if interruption == 'callback' else None)],
                                  show_info=False, log_batch=False)
                self.addCleanup(trainer.close)
                trainer.scaler = torch.amp.GradScaler('cpu', init_scale=128., growth_interval=100)
                task = SeparateParameters(DataLoader(TensorDataset(torch.ones(2), torch.zeros(2)), batch_size=1))
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'interrupt AMP accumulation'):
                        trainer.fit(train_tasks=[task])
                self.assertAlmostEqual(float(model.bad.detach()), .8, places=6)
                self.assertEqual(float(model.good.detach()), 1.)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                self.assertEqual([scheduler.values for scheduler in schedulers], [[1.], []])
                self.assertEqual(trainer.scaler.get_scale(), 128.)
                self.assertEqual(trainer.scaler.state_dict()['_growth_tracker'], 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.fit(train_tasks=[task])
                self.assertAlmostEqual(float(model.bad.detach()), .64, places=6)
                self.assertAlmostEqual(float(model.good.detach()), .8, places=6)
                self.assertEqual(len(schedulers[0].values), 2)
                self.assertAlmostEqual(schedulers[0].values[-1], .64, places=6)
                self.assertEqual(schedulers[1].values, [1.])
                self.assertEqual(trainer.scaler.state_dict()['_growth_tracker'], 2)


@unittest.skipUnless(torch.cuda.is_available(), '需要CUDA')
class CudaAmpClippingTests(unittest.TestCase):
    def test_nonfinite_optimizer_does_not_corrupt_finite_optimizer(self):
        scaler = torch.amp.GradScaler('cuda', init_scale=128.) if hasattr(torch.amp, 'GradScaler') \
            else torch.cuda.amp.GradScaler(init_scale=128.)
        check_nonfinite_clipping(self, 'cuda', scaler)


if __name__ == '__main__':
    unittest.main()
