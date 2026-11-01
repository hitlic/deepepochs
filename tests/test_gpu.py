"""CUDA训练与PyTorch更新对照；无需下载数据。"""
import contextlib
import io
import os
import pickle
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, EpochTask, GradClipCallback, Optimizer, Trainer
from deepepochs import metrics
from deepepochs.callbacks.check import load_state, save_state
from deepepochs.loops import StopLoopException
from deepepochs.patches import ConfusionPatch, MeanPatch, PatchBase, TensorPatch, aggregate_patch_dicts


def fit_quiet(trainer, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return trainer.fit(*args, **kwargs)


def autocast(precision):
    if precision == 'fp32':
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    return torch.autocast(device_type='cuda', dtype=dtype)


def make_scaler(enabled=True):
    if hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_model():
    model = nn.Linear(3, 1).cuda()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.1, 0.2, 0.3]], device='cuda'))
        model.bias.fill_(0.02)
    return model


def reference_step(model, optimizer, inputs, targets, precision, scaler, clip=None):
    optimizer.zero_grad(set_to_none=True)
    with autocast(precision):
        loss = F.mse_loss(model(inputs), targets)
    scaler.scale(loss).backward()
    if clip is not None:
        scaler.unscale_(optimizer)
        clip(model.parameters())
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


class OutputDtypes(Callback):
    def __init__(self):
        super().__init__()
        self.values = []

    def on_after_train_forward(self, trainer, model_out):
        self.values.append(model_out.dtype)


class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = make_model()
        self.right = make_model()
        with torch.no_grad():
            self.right.weight.mul_(0.5)

    def forward(self, inputs):
        return self.left(inputs), self.right(inputs)


class ManualOptimizersTask(EpochTask):
    def train_step(self, batch_x, batch_y):
        first = self.model(batch_x)[0]
        first_loss = self.loss(first, batch_y[:, :1], do_optimize=False, do_metric=False)
        self.optimize(optimize_index=0)
        second = self.model(batch_x)[1]
        second_loss = self.loss(second, batch_y[:, 1:], do_optimize=False, do_metric=False)
        self.optimize(optimize_index=1)
        self.do_metric((first_loss + second_loss) / 2,
                       torch.cat((first, second), dim=1), batch_y)


@unittest.skipUnless(torch.cuda.is_available(), '需要CUDA设备')
class GpuTrainerTests(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 20 - 0.25
        self.targets = self.inputs.sum(1, keepdim=True) * 0.3 + 0.05
        self.cuda_inputs = self.inputs.cuda()
        self.cuda_targets = self.targets.cuda()
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', self.old_tf32)

    def loader(self, batch_size=5, targets=None):
        return DataLoader(TensorDataset(self.inputs, self.targets if targets is None else targets),
                          batch_size=batch_size, shuffle=False)

    def trainer(self, model, precision='fp32', optimizer=None, **kwargs):
        trainer = Trainer(model, F.mse_loss,
                          torch.optim.SGD(model.parameters(), lr=0.05) if optimizer is None else optimizer,
                          epochs=kwargs.pop('epochs', 1), device='cuda', precision=precision,
                          show_info=False, log_batch=False, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def assert_models_close(self, actual, reference, atol=1e-7, rtol=1e-6):
        for parameter, expected in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, atol=atol, rtol=rtol)

    def test_metrics_follow_in_place_cuda_tensor_updates(self):
        matrix = torch.eye(2, dtype=torch.int64, device='cuda')
        self.assertEqual(float(metrics.precision(conf_mat=matrix, average='macro')), 1.)
        matrix[1, 0] = 3
        self.assertEqual(float(metrics.precision(conf_mat=matrix, average='macro')), .625)

        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                predictions = torch.eye(2, dtype=dtype, device='cuda')
                targets = torch.tensor([0, 1], device='cuda')
                self.assertEqual(float(metrics.accuracy(predictions, targets)), 1.)
                predictions.copy_(predictions.flip(1))
                self.assertEqual(float(metrics.accuracy(predictions, targets)), 0.)

                # 复用同一预测与标签对象，分别改变内容后重新计算指标。
                predictions.copy_(predictions.flip(1))
                self.assertEqual(float(metrics.accuracy(predictions, targets)), 1.)
                targets.copy_(targets.flip(0))
                self.assertEqual(float(metrics.accuracy(predictions, targets)), 0.)

    def test_fp32_updates_match_raw_pytorch(self):
        model, reference = make_model(), make_model()
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        scaler = make_scaler(enabled=False)
        for inputs, targets in self.loader(batch_size=2):
            reference_step(reference, optimizer, inputs.cuda(), targets.cuda(), 'fp32', scaler)
        fit_quiet(self.trainer(model), self.loader(batch_size=2))
        self.assert_models_close(model, reference)

    def test_cpu_model_and_adam_move_to_cuda_with_parameter_references_intact(self):
        model = nn.Linear(3, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.1, 0.2, 0.3]]))
            model.bias.fill_(0.02)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.assertTrue(all(parameter.device.type == 'cpu' for parameter in model.parameters()))
        reference = make_model()
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
        scaler = make_scaler(enabled=False)
        for inputs, targets in self.loader(batch_size=2):
            reference_step(reference, reference_optimizer, inputs.cuda(), targets.cuda(), 'fp32', scaler)
        trainer = self.trainer(model, optimizer=optimizer)
        for parameter, optimized in zip(model.parameters(), optimizer.param_groups[0]['params']):
            self.assertEqual(parameter.device.type, 'cuda')
            self.assertIs(optimized, parameter)
        fit_quiet(trainer, self.loader(batch_size=2))
        self.assert_models_close(model, reference)
        for parameter in model.parameters():
            self.assertEqual(optimizer.state[parameter]['exp_avg'].device, parameter.device)

    def check_amp_convergence(self, precision):
        model, reference = make_model(), make_model()
        initial_loss = float(F.mse_loss(model(self.cuda_inputs), self.cuda_targets).detach())
        observer = OutputDtypes()
        trainer = self.trainer(model, precision=precision, epochs=12, callbacks=[observer],
                               optimizer=torch.optim.SGD(model.parameters(), lr=0.2))
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
        scaler = make_scaler(enabled=precision == 'fp16')
        for _ in range(12):
            reference_step(reference, reference_optimizer, self.cuda_inputs,
                           self.cuda_targets, precision, scaler)
        fit_quiet(trainer, self.loader())
        final_loss = float(F.mse_loss(model(self.cuda_inputs), self.cuda_targets).detach())
        self.assertLess(final_loss, initial_loss * 0.9)
        expected_dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
        self.assertEqual(set(observer.values), {expected_dtype})
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        self.assertEqual(trainer.scaler.is_enabled(), precision == 'fp16')
        self.assert_models_close(model, reference, atol=1e-6, rtol=1e-5)

    def test_fp16_autocast_converges_with_fp32_parameters(self):
        self.check_amp_convergence('fp16')

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         '需要支持BF16的CUDA设备')
    def test_bf16_autocast_converges_with_fp32_parameters(self):
        self.check_amp_convergence('bf16')

    def test_norm_and_value_clipping_match_unscaled_reference_gradients(self):
        for precision in ('fp32', 'fp16'):
            for norm_type in (None, 2):
                with self.subTest(precision=precision, norm_type=norm_type):
                    model, reference = make_model(), make_model()
                    maximum = 0.003
                    callback = GradClipCallback(maximum, norm_type=norm_type)
                    trainer = self.trainer(model, precision=precision, callbacks=[callback])
                    optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
                    scaler = make_scaler(enabled=precision == 'fp16')
                    if norm_type is None:
                        clip = lambda parameters: nn.utils.clip_grad_value_(parameters, maximum)
                    else:
                        clip = lambda parameters: nn.utils.clip_grad_norm_(parameters, maximum, norm_type=2)
                    reference_step(reference, optimizer, self.cuda_inputs, self.cuda_targets,
                                   precision, scaler, clip=clip)
                    fit_quiet(trainer, self.loader())
                    self.assert_models_close(model, reference)
                    self.assertFalse(torch.equal(model.weight, make_model().weight))

    def test_accumulation_weights_partial_final_batch_by_actual_samples(self):
        for precision in ('fp32', 'fp16'):
            with self.subTest(precision=precision):
                model, reference = make_model(), make_model()
                trainer = self.trainer(model, precision=precision, gradient_accumulation_steps=3)
                optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
                scaler = make_scaler(enabled=precision == 'fp16')
                reference_step(reference, optimizer, self.cuda_inputs, self.cuda_targets, precision, scaler)
                fit_quiet(trainer, self.loader(batch_size=2))
                if precision == 'fp32':
                    self.assert_models_close(model, reference, atol=3e-8, rtol=1e-6)
                else:
                    self.assert_models_close(model, reference, atol=2e-5, rtol=2e-4)

    def test_interrupted_accumulation_retries_with_fresh_gradients_and_scheduler_loss(self):
        class RecordLoss:
            def __init__(self):
                self.values = []

            def step(self, loss):
                self.values.append(float(loss))

        class Interrupt(Callback):
            def __init__(self, event):
                super().__init__()
                self.event = event
                self.enabled = True

            def stop(self, event):
                if self.enabled and event == self.event:
                    self.enabled = False
                    if event == 'batch':
                        raise StopLoopException('stop incomplete accumulation window')
                    raise RuntimeError('interrupt accumulation window')

            def on_after_train_batch(self, trainer, metrics, batch_idx):
                self.stop('batch')

            def on_after_backward(self, trainer, loss):
                self.stop('backward')

            def on_before_optimize(self, trainer, optimize_index):
                self.stop('optimize')

        inputs, targets = torch.ones(2, 1), torch.tensor([[1.], [3.]])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
        for precision in ('fp32', 'fp16', 'bf16'):
            if precision == 'bf16' and not torch.cuda.is_bf16_supported():
                continue
            for event in ('batch', 'backward', 'optimize'):
                with self.subTest(precision=precision, event=event):
                    model = nn.Linear(1, 1, bias=False).cuda()
                    nn.init.zeros_(model.weight)
                    reference = nn.Linear(1, 1, bias=False).cuda()
                    nn.init.zeros_(reference.weight)
                    scheduler = RecordLoss()
                    raw = torch.optim.SGD(model.parameters(), lr=.1)
                    interrupter = Interrupt(event)
                    trainer = self.trainer(model, precision=precision,
                                           optimizer=Optimizer(raw, scheduler, sched_on='step', sched_with_loss=True),
                                           gradient_accumulation_steps=2, callbacks=[interrupter])
                    if trainer.scaler.is_enabled():
                        trainer.scaler.load_state_dict({'scale': 128., 'growth_factor': 2.,
                                                        'backoff_factor': .5, 'growth_interval': 2,
                                                        '_growth_tracker': 0})
                    initial_scaler = trainer.scaler.state_dict()
                    reference_scaler = make_scaler(enabled=precision == 'fp16')
                    reference_scaler.load_state_dict(initial_scaler)
                    if event == 'batch':
                        fit_quiet(trainer, loader)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'interrupt accumulation window'):
                            fit_quiet(trainer, loader)
                    self.assertEqual(float(model.weight.detach()), 0.)
                    self.assertIsNone(model.weight.grad)
                    self.assertEqual(scheduler.values, [])
                    self.assertEqual(trainer.scaler.state_dict(), initial_scaler)

                    reference_step(reference, torch.optim.SGD(reference.parameters(), lr=.1),
                                   inputs.cuda(), targets.cuda(), precision, reference_scaler)
                    fit_quiet(trainer, loader)
                    self.assert_models_close(model, reference)
                    self.assertAlmostEqual(float(model.weight.detach()), .4, places=6)
                    self.assertEqual(scheduler.values, [5.])
                    self.assertEqual(trainer.scaler.state_dict(), reference_scaler.state_dict())

    def test_interrupted_fp16_manual_optimizers_preserve_committed_update(self):
        class InterruptFirstUpdate(Callback):
            def __init__(self):
                super().__init__()
                self.enabled = True

            def on_after_optimize(self, trainer, optimize_index):
                if self.enabled and optimize_index == 0:
                    self.enabled = False
                    raise RuntimeError('first optimizer already committed')

        targets = torch.cat((self.targets, self.targets * .5), dim=1)
        model, reference = TwoHeads(), TwoHeads()
        optimizers = [torch.optim.SGD(model.left.parameters(), lr=.03),
                      torch.optim.SGD(model.right.parameters(), lr=.07)]
        reference_optimizers = [torch.optim.SGD(reference.left.parameters(), lr=.03),
                                torch.optim.SGD(reference.right.parameters(), lr=.07)]
        trainer = self.trainer(model, precision='fp16', optimizer=optimizers,
                               callbacks=[InterruptFirstUpdate()], gradient_accumulation_steps=2)
        trainer.scaler.load_state_dict({'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
                                        'growth_interval': 2, '_growth_tracker': 0})
        scaler = make_scaler()
        scaler.load_state_dict(trainer.scaler.state_dict())
        # 第一个优化器已完成前四个样本的更新，第二个优化器
        # 在未完成窗口中保留的梯度应丢弃。
        reference_step(reference.left, reference_optimizers[0], self.cuda_inputs[:4],
                       targets[:4, :1].cuda(), 'fp16', scaler)
        task = ManualOptimizersTask(self.loader(batch_size=2, targets=targets))
        with self.assertRaisesRegex(RuntimeError, 'first optimizer already committed'):
            fit_quiet(trainer, train_tasks=[task])
        self.assert_models_close(model, reference, atol=2e-5, rtol=2e-4)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(trainer.scaler.state_dict(), scaler.state_dict())

        # 重试以已提交的参数和scaler状态开始新的累积窗口。
        for start, end in ((0, 4), (4, 5)):
            for index, optimizer in enumerate(reference_optimizers):
                optimizer.zero_grad(set_to_none=True)
                with autocast('fp16'):
                    loss = F.mse_loss(reference(self.cuda_inputs[start:end])[index],
                                      targets[start:end, index:index + 1].cuda())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
            scaler.update()
        fit_quiet(trainer, train_tasks=[task])
        self.assert_models_close(model, reference, atol=3e-5, rtol=1e-3)
        self.assertEqual(trainer.scaler.state_dict(), scaler.state_dict())

    def test_shared_parameters_receive_two_optimizer_updates_without_double_unscale(self):
        for precision in ('fp32', 'fp16'):
            with self.subTest(precision=precision):
                model, reference = make_model(), make_model()
                rates = (0.03, 0.07)
                optimizers = [torch.optim.SGD(model.parameters(), lr=rate) for rate in rates]
                reference_optimizers = [torch.optim.SGD(reference.parameters(), lr=rate) for rate in rates]
                scaler = make_scaler(enabled=precision == 'fp16')
                with autocast(precision):
                    loss = F.mse_loss(reference(self.cuda_inputs), self.cuda_targets)
                scaler.scale(loss).backward()
                # 两个原生优化器使用同一份已反缩放梯度，连续更新参数。
                scaler.unscale_(reference_optimizers[0])
                for optimizer in reference_optimizers:
                    optimizer.step()
                scaler.update()
                fit_quiet(self.trainer(model, precision=precision, optimizer=optimizers), self.loader())
                self.assert_models_close(model, reference)

    def test_manual_optimizer_indices_update_independent_parameter_groups(self):
        targets = torch.cat((self.targets, self.targets * 0.5), dim=1)
        for precision in ('fp32', 'fp16'):
            with self.subTest(precision=precision):
                model, reference = TwoHeads(), TwoHeads()
                optimizers = [torch.optim.SGD(model.left.parameters(), lr=0.03),
                              torch.optim.SGD(model.right.parameters(), lr=0.07)]
                reference_optimizers = [torch.optim.SGD(reference.left.parameters(), lr=0.03),
                                        torch.optim.SGD(reference.right.parameters(), lr=0.07)]
                scaler = make_scaler(enabled=precision == 'fp16')
                for index, optimizer in enumerate(reference_optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(precision):
                        output = reference(self.cuda_inputs)[index]
                        loss = F.mse_loss(output, targets[:, index:index + 1].cuda())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                scaler.update()
                trainer = self.trainer(model, precision=precision, optimizer=optimizers)
                task = ManualOptimizersTask(self.loader(targets=targets))
                fit_quiet(trainer, train_tasks=[task])
                self.assert_models_close(model, reference)

    def test_checkpoint_restores_scaler_scheduler_and_next_update(self):
        model = make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        trainer = self.trainer(model, precision='fp16',
                               optimizer=Optimizer(optimizer, scheduler, sched_on='step'))
        trainer.scaler.load_state_dict({'scale': 128., 'growth_factor': 2.,
                                        'backoff_factor': 0.5, 'growth_interval': 2, '_growth_tracker': 0})
        fit_quiet(trainer, self.loader())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'checkpoint.ckpt')
            save_state(trainer, trainer.model, trainer.opt, path, epoch=0)
            restored_model = make_model()
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.05, momentum=0.5)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.7)
            restored = self.trainer(restored_model, precision='fp16', epochs=2,
                                    optimizer=Optimizer(restored_optimizer, restored_scheduler, sched_on='step'))
            load_state(restored, path)
        self.assertEqual(restored.init_epoch, 1)
        self.assertEqual(restored.scaler.state_dict(), trainer.scaler.state_dict())
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(restored_optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
        trainer.max_epochs = 2
        fit_quiet(trainer, self.loader())
        fit_quiet(restored, self.loader())
        self.assert_models_close(restored_model, model, atol=0, rtol=0)
        self.assertEqual(restored.scaler.state_dict(), trainer.scaler.state_dict())
        self.assertEqual(restored.scaler.get_scale(), 256.)
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_fp16_overflow_skips_optimizer_and_step_scheduler(self):
        model, reference = make_model(), make_model()
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.5)
        scaler = make_scaler()
        with autocast('fp16'):
            loss = F.mse_loss(reference(self.cuda_inputs), self.cuda_targets) * 1e10
        scaler.scale(loss).backward()
        reference_steps = []
        handle = reference_optimizer.register_step_post_hook(lambda *args: reference_steps.append(True))
        try:
            scaler.step(reference_optimizer)
        finally:
            handle.remove()
        if reference_steps:
            reference_scheduler.step()
        scaler.update()
        trainer = Trainer(model, lambda preds, targets: F.mse_loss(preds, targets) * 1e10,
                          Optimizer(optimizer, scheduler, sched_on='step'), epochs=1,
                          device='cuda', precision='fp16', show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        previous_scale = trainer.scaler.get_scale()
        previous_epoch = scheduler.last_epoch
        fit_quiet(trainer, self.loader())
        self.assertEqual(reference_steps, [])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
        self.assert_models_close(model, reference, atol=0, rtol=0)
        self.assertEqual(scheduler.last_epoch, previous_epoch)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.05)
        self.assertLess(trainer.scaler.get_scale(), previous_scale)
        self.assertEqual(trainer.scaler.state_dict(), scaler.state_dict())
        self.assertEqual(scheduler.state_dict(), reference_scheduler.state_dict())

    def test_amp_scheduler_follows_optimizer_after_compile(self):
        class ObserveClippedGradients(Callback):
            def __init__(self):
                super().__init__(priority=2)
                self.finite = None

            def on_before_optimize(self, trainer, optimize_index):
                self.finite = all(bool(torch.isfinite(parameter.grad).all())
                                  for parameter in trainer.model.parameters() if parameter.grad is not None)

        # 编译执行和随后普通Trainer的AMP更新可以发生在同一进程。
        compiled = torch.compile(nn.Linear(1, 1), backend='eager')
        compiled(torch.ones(1, 1))
        loader = DataLoader(TensorDataset(self.inputs[:1], self.targets[:1]), batch_size=1)
        for overflow in (False, True):
            with self.subTest(overflow=overflow):
                model = make_model()
                initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                raw = torch.optim.SGD(model.parameters(), lr=.05)
                scheduler = torch.optim.lr_scheduler.StepLR(raw, step_size=1, gamma=.5)
                observer = ObserveClippedGradients()
                trainer = self.trainer(model, precision='fp16',
                                       optimizer=Optimizer(raw, scheduler, sched_on='step'),
                                       callbacks=[GradClipCallback(.003), observer])
                trainer.scaler.load_state_dict({'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
                                                'growth_interval': 2000, '_growth_tracker': 0})
                if overflow:
                    trainer.loss.loss_fn = lambda preds, targets: F.mse_loss(preds, targets) * 1e10
                fit_quiet(trainer, loader)
                self.assertTrue(observer.finite)
                if overflow:
                    # value裁剪可将Inf梯度改为有限值，但AMP本轮仍应跳过更新。
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
                    self.assertEqual(scheduler.last_epoch, 0)
                    self.assertEqual(raw.param_groups[0]['lr'], .05)
                    self.assertEqual(trainer.scaler.get_scale(), 64.)
                else:
                    self.assertTrue(any(not torch.equal(parameter, initial[name])
                                        for name, parameter in model.named_parameters()))
                    self.assertEqual(scheduler.last_epoch, 1)
                    self.assertEqual(raw.param_groups[0]['lr'], .025)
                    self.assertEqual(trainer.scaler.get_scale(), 128.)

    def test_distributed_patch_state_restores_original_cuda_devices(self):
        weights = torch.tensor([1., 2.], device='cuda')

        def tensor_metric(preds, targets):
            self.assertEqual(preds.device, weights.device)
            self.assertEqual(targets.device, weights.device)
            sample_weights = weights[targets]
            return ((preds.argmax(1) == targets) * sample_weights).sum() / sample_weights.sum()

        def confusion_metric(matrix):
            self.assertEqual(matrix.device, weights.device)
            return (matrix.diag() * weights).sum() / (matrix.sum(1) * weights).sum()

        class CudaMeanPatch(MeanPatch):
            def forward(self):
                self_test.assertEqual(self.batch_value.device, weights.device)
                self_test.assertEqual(self.batch_size.device, weights.device)
                return super().forward()

        class CudaCountPatch(PatchBase):
            def __init__(self):
                super().__init__()
                self.counts = {'total': torch.tensor(0., device='cuda')}

            def load(self, counts, targets=None):
                self.counts = counts
                return self

            def forward(self):
                self_test.assertEqual(self.counts['total'].device, weights.device)
                return self.counts['total']

            def add(self, other):
                self.counts = {'total': self.counts['total'] + other.counts['total']}
                return self

        self_test = self
        predictions = torch.tensor([[0.8, 0.2], [0.2, 0.8]], device='cuda')
        targets = torch.tensor([0, 1], device='cuda')
        local = {
            'tensor': TensorPatch(tensor_metric).load(predictions, targets),
            'confusion': ConfusionPatch(confusion_metric).load(predictions, targets),
            'mean': CudaMeanPatch(lambda preds, labels: preds.mean(),
                                  torch.tensor(2, device='cuda')).load(predictions, targets),
            'custom': CudaCountPatch().load({'total': torch.tensor(2., device='cuda')}),
        }
        remote = {
            'tensor': {
                'batch_preds': [[torch.tensor([[0.9, 0.1]])],
                                [torch.tensor([[0.1, 0.9], [0.7, 0.3]])]],
                'batch_targets': [[torch.tensor([1])], [torch.tensor([1, 0])]],
            },
            'confusion': {'confusion_matrix': torch.tensor([[1, 0], [1, 1]])},
            'mean': {'batch_size': torch.tensor(3), 'batch_value': torch.tensor(4.)},
            'custom': {'counts': {'total': torch.tensor(3.)}},
        }

        class Context:
            is_distributed = True
            rank = 0
            world_size = 2

            def all_gather_object(self, data):
                pickle.dumps(data)
                self_test.assertEqual(data['tensor']['batch_preds'][0][0].device.type, 'cpu')
                self_test.assertEqual(data['tensor']['batch_targets'][0][0].device.type, 'cpu')
                self_test.assertNotIn('metric', data['tensor'])
                return [data, remote]

            def broadcast_object(self, data, src):
                self_test.assertEqual(src, 0)
                return data

        actual = aggregate_patch_dicts([local], Context())
        self.assertEqual(float(actual['tensor']), 0.75)
        self.assertEqual(float(actual['confusion']), 0.75)
        self.assertEqual(float(actual['mean']), 1.)
        self.assertEqual(float(actual['custom']), 5.)

    def test_two_updates_in_one_custom_step_follow_scaler_growth(self):
        class RepeatedUpdateTask(EpochTask):
            def train_step(self, batch_x, batch_y):
                self.loss(self.model(batch_x), batch_y)
                self.loss(self.model(batch_x), batch_y)

        model, reference = make_model(), make_model()
        trainer = self.trainer(model, precision='fp16')
        scaler_state = {'scale': 128., 'growth_factor': 2., 'backoff_factor': 0.5,
                        'growth_interval': 1, '_growth_tracker': 0}
        trainer.scaler.load_state_dict(scaler_state)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        scaler = make_scaler()
        scaler.load_state_dict(scaler_state)
        for _ in range(2):
            reference_step(reference, optimizer, self.cuda_inputs, self.cuda_targets, 'fp16', scaler)
        fit_quiet(trainer, train_tasks=[RepeatedUpdateTask(self.loader())])
        self.assert_models_close(model, reference)
        self.assertEqual(trainer.scaler.get_scale(), 512.)
        self.assertEqual(trainer.scaler.state_dict(), scaler.state_dict())

    def test_optimizer_loss_windows_stay_separate_with_amp(self):
        class RecordLoss:
            def __init__(self): self.values = []
            def step(self, loss): self.values.append(float(loss))

        class Task(EpochTask):
            def train_step(self, batch_x, batch_y):
                self.loss(self.model(batch_x)[0], batch_y[:, :1], do_optimize=False, do_metric=False)
                self.optimize(0)
                self.loss(self.model(batch_x)[1], batch_y[:, 1:], do_optimize=False, do_metric=False)
                self.optimize(1)

        for precision in ('fp32', 'fp16'):
            with self.subTest(precision=precision):
                model = TwoHeads()
                with torch.no_grad():
                    for parameter in model.parameters(): parameter.zero_()
                schedulers = [RecordLoss(), RecordLoss()]
                optimizers = [Optimizer(torch.optim.SGD(head.parameters(), lr=0.), schedulers[i],
                                        sched_on='step', sched_with_loss=True)
                              for i, head in enumerate((model.left, model.right))]
                trainer = self.trainer(model, precision=precision, optimizer=optimizers,
                                       gradient_accumulation_steps=2)
                if trainer.scaler.is_enabled():
                    trainer.scaler.load_state_dict({'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
                                                    'growth_interval': 2000, '_growth_tracker': 0})
                loader = DataLoader(TensorDataset(self.inputs[:2], torch.tensor([[1., 10.], [3., 30.]])),
                                    batch_size=1)
                fit_quiet(trainer, train_tasks=[Task(loader)])
                self.assertEqual(schedulers[0].values, [5.])
                self.assertEqual(schedulers[1].values, [500.])


if __name__ == '__main__':
    unittest.main()
