"""小张量训练参考：比较真实更新和统计值，不依赖数据下载。"""
import contextlib
import io
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, EpochTask, GradClipCallback, LogCallback, Optimizer, Trainer
from deepepochs.loops import StopLoopException


def fit_quiet(trainer, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return trainer.fit(*args, **kwargs)


def make_model():
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.1, .2, .3]]))
    return model


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
        self.y = self.x.sum(1, keepdim=True)

    def trainer(self, model, **kwargs):
        return Trainer(model, nn.functional.mse_loss,
                       torch.optim.SGD(model.parameters(), lr=.01),
                       device='cpu', show_info=False, log_batch=False, **kwargs)

    def test_standard_update_matches_pytorch(self):
        model, reference = make_model(), make_model()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        dl = DataLoader(TensorDataset(self.x, self.y), batch_size=2)
        losses = []
        for x, y in dl:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(reference(x), y)
            losses.append((loss.item(), len(x)))
            loss.backward()
            optimizer.step()
        result = fit_quiet(self.trainer(model, epochs=1), dl)
        torch.testing.assert_close(model.weight, reference.weight)
        self.assertIsInstance(result['train']['loss'][0], np.ndarray)
        expected = sum(loss * count for loss, count in losses) / len(self.x)
        self.assertAlmostEqual(float(result['train']['loss'][0]), expected, places=6)

    def test_multiple_optimizers_keep_shared_parameter_updates(self):
        model, reference = make_model(), make_model()
        opts = [torch.optim.SGD(reference.parameters(), lr=.01) for _ in range(2)]
        loss = nn.functional.mse_loss(reference(self.x), self.y)
        loss.backward()
        for opt in opts:
            opt.step()
        trainer = Trainer(model, nn.functional.mse_loss,
                          [torch.optim.SGD(model.parameters(), lr=.01) for _ in range(2)],
                          epochs=1, device='cpu', show_info=False, log_batch=False)
        fit_quiet(trainer, DataLoader(TensorDataset(self.x, self.y), batch_size=5))
        torch.testing.assert_close(model.weight, reference.weight)

    def test_custom_task_return_patch_and_step_arguments(self):
        from deepepochs import ValuePatch
        class Task(EpochTask):
            def train_step(self, batch_x, batch_y, number):
                self.loss(self.model(batch_x), batch_y)
                return {'custom': ValuePatch(len(batch_x)).load(number)}
        trainer = self.trainer(make_model(), epochs=1)
        result = fit_quiet(trainer, train_tasks=[Task(
            DataLoader(TensorDataset(self.x, self.y), batch_size=2), number=7)])
        self.assertEqual(float(result['train']['custom'][0]), 7)

class TrainerCorrectnessTests(unittest.TestCase):
    setUp = TrainerTests.setUp
    trainer = TrainerTests.trainer
    def test_gradient_clipping_precedes_update(self):
        model, reference = make_model(), make_model()
        raw = torch.optim.SGD(reference.parameters(), lr=.01)
        nn.functional.mse_loss(reference(self.x * 100), self.y).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
        raw.step()
        trainer = self.trainer(model, epochs=1, callbacks=[GradClipCallback(.1, norm_type=2)])
        fit_quiet(trainer, DataLoader(TensorDataset(self.x * 100, self.y), batch_size=5))
        torch.testing.assert_close(model.weight, reference.weight)

    def test_callable_batch_size_keeps_single_sample_dimension(self):
        model = make_model()
        trainer = self.trainer(model)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.test(DataLoader(TensorDataset(self.x, self.y), batch_size=2),
                                  batch_size=lambda x, y: x.shape[0])
        self.assertAlmostEqual(float(result['loss']), nn.functional.mse_loss(model(self.x), self.y).item(), places=6)

    def test_dictionary_model_computes_own_loss(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = make_model()
            def forward(self, x, y):
                return {'loss': nn.functional.mse_loss(self.linear(x), y)}
        model = Model()
        trainer = Trainer(model, opt=torch.optim.SGD(model.parameters(), lr=.01), device='cpu',
                          epochs=1, show_info=False, log_batch=False)
        loader = DataLoader([{'x': x, 'y': y} for x, y in zip(self.x, self.y)], batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.test(loader)
        self.assertAlmostEqual(float(result['loss']), nn.functional.mse_loss(model.linear(self.x), self.y).item(), places=6)
        self.assertEqual(len(fit_quiet(trainer, loader)['train']['loss']), 1)

    def test_epoch_plateau_uses_training_loss_when_validation_loss_disabled(self):
        model = make_model()
        raw = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(raw)
        trainer = Trainer(model, nn.functional.mse_loss, Optimizer(raw, scheduler, sched_with_loss=True),
                          epochs=1, device='cpu', show_info=False, log_batch=False)
        def metric(preds, targets):
            return (preds - targets).abs().mean()
        dl = DataLoader(TensorDataset(self.x, self.y), batch_size=2)
        fit_quiet(trainer, dl, dl, do_val_loss=False, val_metrics=[metric])
        self.assertEqual(scheduler.last_epoch, 1)

    def test_step_plateau_receives_current_loss(self):
        model = make_model()
        raw = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(raw)
        trainer = Trainer(model, nn.functional.mse_loss,
                          Optimizer(raw, scheduler, sched_on='step', sched_with_loss=True),
                          epochs=1, device='cpu', show_info=False, log_batch=False)
        fit_quiet(trainer, DataLoader(TensorDataset(self.x, self.y), batch_size=2))
        self.assertEqual(scheduler.last_epoch, 3)

    def test_reused_task_does_not_mix_phase_metrics(self):
        def train_only(preds, targets): return torch.tensor(1.)
        def val_only(preds, targets): return torch.tensor(2.)
        task = EpochTask(DataLoader(TensorDataset(self.x, self.y), batch_size=2))
        trainer = self.trainer(make_model(), epochs=2)
        result = fit_quiet(trainer, train_tasks=[task], val_tasks=[task],
                           train_metrics=[train_only], val_metrics=[val_only])
        self.assertNotIn('val_only', result['train'])
        self.assertNotIn('train_only', result['val'])
        self.assertEqual(task.metrics, [])

    def test_reused_validation_task_list_keeps_current_loader_loss_and_scheduler(self):
        observed_tasks, scheduler_losses = [], []

        class Recorder(Callback):
            def on_before_val_epoch(self, trainer, task):
                observed_tasks.append(task)

        class Scheduler:
            def step(self, loss):
                scheduler_losses.append(float(loss))

        def auxiliary(preds, targets):
            return torch.tensor(7.)

        inputs = torch.zeros(1, 1)
        train_dl = DataLoader(TensorDataset(inputs, inputs), batch_size=1)
        val_dls = [DataLoader(TensorDataset(inputs, torch.full_like(inputs, target)), batch_size=1)
                   for target in (1., 3.)]
        task = EpochTask(train_dl, metrics=[auxiliary], do_loss=False)
        tasks = [task]
        model = nn.Linear(1, 1, bias=False)
        # 零输入让模型输出和训练梯度保持为零，两个验证集的参考损失分别为1、9。
        optimizer = Optimizer(torch.optim.SGD(model.parameters(), lr=.1), Scheduler(),
                              sched_with_loss=True)
        trainer = Trainer(model, nn.functional.mse_loss, optimizer, epochs=1,
                          callbacks=[Recorder()], device='cpu', show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        first = fit_quiet(trainer, train_dl, val_dls[0], val_tasks=tasks)
        second = fit_quiet(trainer, train_dl, val_dls[1], val_tasks=tasks, epochs=1)

        self.assertEqual([float(first['val']['loss'][0]), float(second['val']['loss'][0])], [1., 9.])
        self.assertEqual(scheduler_losses, [1., 9.])
        self.assertEqual([float(first['val']['auxiliary'][0]), float(second['val']['auxiliary'][0])], [7., 7.])
        self.assertEqual(len(tasks), 1)
        self.assertIs(tasks[0], task)
        self.assertEqual(len(observed_tasks), 4)
        self.assertIs(observed_tasks[0].dataloader, val_dls[0])
        self.assertIs(observed_tasks[1], task)
        self.assertIs(observed_tasks[2].dataloader, val_dls[1])
        self.assertIs(observed_tasks[3], task)

    def test_reused_test_task_list_keeps_current_loader_loss_and_order(self):
        observed_tasks = []

        class Recorder(Callback):
            def on_before_test_epoch(self, trainer, task):
                observed_tasks.append(task)

        def auxiliary(preds, targets):
            return torch.tensor(7.)

        inputs = torch.zeros(1, 1)
        test_dls = [DataLoader(TensorDataset(inputs, torch.full_like(inputs, target)), batch_size=1)
                    for target in (1., 3.)]
        task = EpochTask(test_dls[0], metrics=[auxiliary], do_loss=False)
        tasks = [task]
        model = nn.Linear(1, 1, bias=False)
        trainer = self.trainer(model, callbacks=[Recorder()])
        self.addCleanup(trainer.close)
        with contextlib.redirect_stdout(io.StringIO()):
            first = trainer.test(test_dls[0], tasks=tasks)
            second = trainer.test(test_dls[1], tasks=tasks)

        self.assertEqual([float(first['loss']), float(second['loss'])], [1., 9.])
        self.assertEqual([float(first['auxiliary']), float(second['auxiliary'])], [7., 7.])
        self.assertEqual(len(tasks), 1)
        self.assertIs(tasks[0], task)
        self.assertEqual(len(observed_tasks), 4)
        self.assertIs(observed_tasks[0].dataloader, test_dls[0])
        self.assertIs(observed_tasks[1], task)
        self.assertIs(observed_tasks[2].dataloader, test_dls[1])
        self.assertIs(observed_tasks[3], task)

    def test_bf16_metric_can_be_returned_as_numpy(self):
        model = make_model()
        trainer = Trainer(model, nn.functional.mse_loss, device='cpu', precision='bf16',
                          show_info=False, log_batch=False)
        def mean_prediction(preds, targets):
            return preds.mean()
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.test(DataLoader(TensorDataset(self.x, self.y), batch_size=2), metrics=[mean_prediction])
        self.assertEqual(result['mean_prediction'].dtype, np.float32)

    def test_linear_cosine_schedule_initializes_and_warms_up(self):
        from deepepochs import LinearCosineScheduler
        parameter = nn.Parameter(torch.ones(1))
        raw = torch.optim.SGD([parameter], lr=.01)
        scheduler = LinearCosineScheduler(raw, .1, 2, 6, min_val=0.)
        self.assertAlmostEqual(raw.param_groups[0]['lr'], .05)
        raw.step()
        scheduler.step()
        self.assertAlmostEqual(raw.param_groups[0]['lr'], .1)
        for _ in range(5):
            raw.step()
            scheduler.step()
        self.assertAlmostEqual(raw.param_groups[0]['lr'], 0.)


class TrainerLifecycleTests(unittest.TestCase):
    setUp = TrainerTests.setUp
    trainer = TrainerTests.trainer

    def test_explicit_retry_epochs_start_from_completed_progress(self):
        for phase in ('batch', 'epoch'):
            for error_type in (StopLoopException, RuntimeError):
                with self.subTest(phase=phase, error=error_type.__name__):
                    class StopOnce(Callback):
                        enabled = True

                        def stop(self):
                            if self.enabled:
                                self.enabled = False
                                raise error_type('interrupt once')

                        def on_after_train_batch(self, trainer, metrics, batch_idx):
                            if phase == 'batch':
                                self.stop()

                        def on_after_epoch(self, trainer, **kwargs):
                            if phase == 'epoch':
                                self.stop()

                    trainer = self.trainer(make_model(), epochs=5, callbacks=[StopOnce()])
                    loader = DataLoader(TensorDataset(self.x, self.y), batch_size=5)
                    try:
                        if error_type is RuntimeError:
                            with self.assertRaisesRegex(RuntimeError, 'interrupt once'):
                                fit_quiet(trainer, loader)
                        else:
                            fit_quiet(trainer, loader)
                        completed = 0 if phase == 'batch' else 1
                        self.assertEqual(trainer.init_epoch, completed)
                        result = fit_quiet(trainer, loader, epochs=1)
                        self.assertEqual(len(result['train']['loss']), 1)
                        self.assertEqual(trainer.init_epoch, completed + 1)
                    finally:
                        trainer.close()


    def test_loss_denominator_matches_full_batch_accumulation(self):
        inputs = torch.tensor([[0, 1, 2], [2, 0, 1]])
        targets = torch.tensor([[0, -100, -100], [1, 0, 1]])
        for weights in (None, torch.tensor([.5, 2.])):
            with self.subTest(weighted=weights is not None):
                model = nn.Embedding(3, 2)
                reference = nn.Embedding(3, 2)
                with torch.no_grad():
                    model.weight.copy_(torch.tensor([[.1, .2], [.3, -.1], [-.2, .4]]))
                    reference.weight.copy_(model.weight)

                def loss(preds, labels):
                    return nn.functional.cross_entropy(preds.transpose(1, 2), labels,
                                                       ignore_index=-100, weight=weights)

                def count(batch_x, batch_y):
                    valid = batch_y[batch_y != -100]
                    return valid.numel() if weights is None else weights[valid].sum().item()

                raw = torch.optim.SGD(reference.parameters(), lr=.1)
                expected_loss = loss(reference(inputs), targets)
                expected_loss.backward()
                raw.step()
                with Trainer(model, loss, torch.optim.SGD(model.parameters(), lr=.1),
                             epochs=1, device='cpu', gradient_accumulation_steps=2,
                             show_info=False, log_batch=False) as trainer:
                    result = fit_quiet(trainer, DataLoader(TensorDataset(inputs, targets), batch_size=1),
                                       batch_size=count)
                torch.testing.assert_close(model.weight, reference.weight)
                self.assertAlmostEqual(float(result['train']['loss'][0]), float(expected_loss), places=6)

    def test_retry_without_epochs_keeps_existing_target(self):
        class StopOnce(Callback):
            enabled = True

            def on_after_epoch(self, trainer, **kwargs):
                if self.enabled:
                    self.enabled = False
                    raise StopLoopException('interrupt once')

        trainer = self.trainer(make_model(), epochs=3, callbacks=[StopOnce()])
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=5)
        try:
            fit_quiet(trainer, loader)
            self.assertEqual(trainer.init_epoch, 1)
            result = fit_quiet(trainer, loader)
            self.assertEqual(len(result['train']['loss']), 2)
            self.assertEqual(trainer.init_epoch, 3)
        finally:
            trainer.close()

    def test_context_closes_writer_on_normal_and_exception_exit(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                class CaptureWriter(Callback):
                    def on_before_train_epoch(self, trainer, task):
                        self.writer = trainer.logger
                        self.worker = self.writer.file_writer.event_writer._async_writer._worker

                    def on_after_train_batch(self, trainer, metrics, batch_idx):
                        if fail:
                            raise RuntimeError('training failed')

                log = LogCallback(log_dir=directory)
                capture = CaptureWriter()
                trainer = self.trainer(make_model(), epochs=1, callbacks=[log, capture])
                expected = self.assertRaisesRegex(RuntimeError, 'training failed') if fail else contextlib.nullcontext()
                with expected:
                    with trainer as active:
                        self.assertIs(active, trainer)
                        fit_quiet(active, DataLoader(TensorDataset(self.x, self.y), batch_size=5))
                self.assertFalse(capture.worker.is_alive())
                self.assertIsNone(trainer.logger)
                self.assertIsNone(log.logger)
                self.assertIsNone(capture.writer.file_writer)


if __name__ == '__main__':
    unittest.main()
