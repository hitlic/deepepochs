"""小张量训练参考：比较真实更新和统计值，不依赖数据下载。"""
import contextlib
import io
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, EpochTask, GradClipCallback, Optimizer, Trainer


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


if __name__ == '__main__':
    unittest.main()
