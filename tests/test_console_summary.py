"""单行轮次摘要、逐批指标与原生训练更新保持一致。"""
import contextlib
import io
import os
import re
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import EpochTask, Optimizer, Trainer


class TerminalStream(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 1


def make_model():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2, .3]]))
        model.bias.fill_(.1)
    return model


def mae(predictions, targets):
    return (predictions - targets).abs().mean()


class ConsoleSummaryTests(unittest.TestCase):
    def loader(self):
        inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
        targets = inputs.sum(1, keepdim=True)
        return DataLoader(TensorDataset(inputs, targets), batch_size=2)

    def optimizer(self, model):
        return torch.optim.SGD([{'params': [model.weight], 'lr': .1},
                                {'params': [model.bias], 'lr': .01}], momentum=.6)

    def test_two_epoch_summary_preserves_updates_metrics_and_scheduler_order(self):
        model, reference = make_model(), make_model()
        raw, native = self.optimizer(model), self.optimizer(reference)
        scheduler = torch.optim.lr_scheduler.StepLR(raw, step_size=1, gamma=.5)
        native_scheduler = torch.optim.lr_scheduler.StepLR(native, step_size=1, gamma=.5)
        loader = self.loader()
        expected_train, expected_val = [], []
        for _ in range(2):
            losses = []
            for inputs, targets in loader:
                native.zero_grad()
                loss = nn.functional.mse_loss(reference(inputs), targets)
                losses.append((loss.item(), len(inputs)))
                loss.backward()
                native.step()
            expected_train.append(sum(value * count for value, count in losses) / len(loader.dataset))
            inputs, targets = loader.dataset.tensors
            expected_val.append(nn.functional.mse_loss(reference(inputs), targets).item())
            native_scheduler.step()

        output = io.StringIO()
        with Trainer(model, nn.functional.mse_loss, Optimizer(raw, scheduler),
                     device='cpu', epochs=2, metrics=[mae], log_batch=False) as trainer:
            with contextlib.redirect_stdout(output), patch(
                    'deepepochs.callbacks.default.perf_counter', side_effect=[10., 11., 20., 22.]):
                history = trainer.fit(loader, loader)

        text = output.getvalue()
        self.assertNotIn('\r', text)
        self.assertNotIn('\x1b', text)
        startup = [line for line in text.splitlines() if line.startswith('init lr:')]
        self.assertEqual(len(startup), 1)
        self.assertIn('opt0/group0=1.000e-01 opt0/group1=1.000e-02', startup[0])
        self.assertFalse(any(line.startswith('lr:') for line in text.splitlines()))
        lines = [line for line in text.splitlines() if line.startswith('E ')]
        self.assertEqual(len(lines), 2)
        for index, line in enumerate(lines, start=1):
            self.assertRegex(line, rf'^E\s+{index}/2\s+TRAIN>')
            self.assertLess(line.index('TRAIN>'), line.index('VAL>'))
            self.assertNotIn('lr:', line)
            self.assertNotIn('time:', line)
            self.assertEqual(line.count('loss:'), 2)
            self.assertEqual(line.count('mae:'), 2)
            self.assertRegex(line, rf'\S {{3}}\[{index}\.00s\]$')
        self.assertEqual([group['lr'] for group in raw.param_groups],
                         [group['lr'] for group in native.param_groups])
        for key, expected in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], expected)
        for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(raw.state[actual_parameter]['momentum_buffer'],
                                       native.state[expected_parameter]['momentum_buffer'])
        self.assertEqual(scheduler.state_dict(), native_scheduler.state_dict())
        for actual, expected in zip(history['train']['loss'], expected_train):
            self.assertAlmostEqual(float(actual), expected, places=6)
        for actual, expected in zip(history['val']['loss'], expected_val):
            self.assertAlmostEqual(float(actual), expected, places=6)

    def test_default_terminal_updates_every_batch_with_loss_and_named_metric(self):
        model = make_model()
        loader = self.loader()
        output = TerminalStream()
        with Trainer(model, nn.functional.mse_loss, torch.optim.SGD(model.parameters(), lr=0.),
                     device='cpu', epochs=1, metrics=[mae], show_info=False) as trainer:
            with contextlib.redirect_stdout(output), patch.dict('os.environ', {'NO_COLOR': '1'}), \
                    patch('deepepochs.loops.os.get_terminal_size', return_value=os.terminal_size((240, 60))):
                history = trainer.fit(train_tasks=[EpochTask(loader, name='domain')])
        text = output.getvalue()
        self.assertNotRegex(text, r'\x1b\[[0-9;]*m')
        batches = re.findall(r'E\s+1/1\s+B\s+\d+/3\s+TRAIN>[^\r\n]*', text)
        self.assertEqual(len(batches), 3)
        for index, batch in enumerate(batches, start=1):
            self.assertRegex(batch, rf'B\s+{index}/3\s+TRAIN>')
            self.assertIn('domain/loss:', batch)
            self.assertIn('domain/mae:', batch)
        summaries = re.findall(r'E\s+1/1\s+TRAIN>[^\r\n]*', text)
        self.assertEqual(len(summaries), 1)
        self.assertIn('domain/loss:', summaries[0])
        self.assertIn('domain/mae:', summaries[0])
        inputs, targets = loader.dataset.tensors
        self.assertAlmostEqual(float(history['train']['domain/loss'][0]),
                               nn.functional.mse_loss(model(inputs), targets).item(), places=6)

    def test_each_test_task_has_its_own_summary_time_and_empty_batch_progress(self):
        def score(predictions, targets):
            return predictions.mean()

        tasks = []
        for name, start in (('left', 1), ('right', 4)):
            inputs = torch.arange(start, start + 3, dtype=torch.float32).reshape(-1, 1)
            loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
            tasks.append(EpochTask(loader, name=name, do_loss=False))
        output = io.StringIO()
        with Trainer(nn.Identity(), nn.functional.mse_loss, device='cpu', metrics=[score],
                     compute_batch_metrics=False, show_info=False) as trainer:
            with contextlib.redirect_stdout(output), patch(
                    'deepepochs.callbacks.default.perf_counter', side_effect=[100., 101., 200., 202.]):
                result = trainer.test(tasks=tasks)
        text = output.getvalue()
        self.assertNotIn('\r', text)
        lines = [line for line in text.splitlines() if line.startswith('E ')]
        batches = [line for line in lines if re.search(r'\bB\s+\d+/', line)]
        summaries = [line for line in lines if line not in batches]
        self.assertEqual(len(batches), 4)
        self.assertTrue(all('TEST>' in line for line in batches))
        self.assertEqual(len(summaries), 2)
        self.assertIn('left/score:', summaries[0])
        self.assertIn('right/score:', summaries[1])
        self.assertRegex(summaries[0], r'\S {3}\[1\.00s\]$')
        self.assertRegex(summaries[1], r'\S {3}\[2\.00s\]$')
        self.assertEqual({key: float(value) for key, value in result.items()},
                         {'left/score': 2., 'right/score': 5.})


if __name__ == '__main__':
    unittest.main()
