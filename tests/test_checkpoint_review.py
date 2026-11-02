"""Checkpoint监控、恢复及临时测试模型的回归。"""
import contextlib
from copy import deepcopy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import CheckCallback, Optimizer, Trainer
from deepepochs.callbacks.check import (
    finalize_checkpoints, get_latest_running, load_state, restore_fit_state, restore_test_model,
)
from deepepochs.loops import StopLoopException


class CheckpointReviewTests(unittest.TestCase):
    def trainer(self, callbacks=None, **kwargs):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer = Trainer(model, nn.functional.mse_loss, Optimizer(optimizer, scheduler),
                          device='cpu', show_info=False, log_batch=False,
                          callbacks=callbacks, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def loader(self):
        return DataLoader(TensorDataset(torch.tensor([[1.], [2.]]),
                                        torch.tensor([[2.], [4.]])), batch_size=2)

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for item, other in zip(actual, expected):
                self.assert_state_equal(item, other)
        else:
            self.assertEqual(actual, expected)

    def test_testing_best_weights_restores_current_training_state_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([callback], running_id='current')
            self.quiet(trainer.fit, self.loader(), epochs=1)
            best = deepcopy(trainer.model.state_dict())
            # 当前模型和动量继续向前，最佳文件保留上一轮。
            with torch.no_grad():
                trainer.model.weight.fill_(0.9)
            trainer.init_epoch = 3
            trainer.opt[0].opt.state[trainer.model.weight]['momentum_buffer'].fill_(0.7)
            expected_model = deepcopy(trainer.model.state_dict())
            expected_opt = deepcopy(trainer.opt.state_dict())
            expected_scaler = deepcopy(trainer.scaler.state_dict())
            observed = []

            def record_weight(preds, targets):
                observed.append(trainer.model.weight.detach().clone())
                return nn.functional.mse_loss(preds, targets)

            self.quiet(trainer.test, self.loader(), metrics=[record_weight])
            self.assertTrue(observed)
            for weight in observed:
                torch.testing.assert_close(weight, best['weight'])
            self.assert_state_equal(trainer.model.state_dict(), expected_model)
            self.assert_state_equal(trainer.opt.state_dict(), expected_opt)
            self.assertEqual(trainer.init_epoch, 3)
            self.assertEqual(trainer.scaler.state_dict(), expected_scaler)
            reference = self.trainer()
            reference.model.load_state_dict(expected_model)
            reference.opt.load_state_dict(expected_opt)
            reference.init_epoch = 3
            trainer.max_epochs = reference.max_epochs = 4
            self.quiet(trainer.fit, self.loader())
            self.quiet(reference.fit, self.loader())
            self.assert_state_equal(trainer.model.state_dict(), reference.model.state_dict())
            self.assert_state_equal(trainer.opt.state_dict(), reference.opt.state_dict())

    def test_failed_test_restores_current_model_in_finally(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([callback])
            callback.do_check(trainer, {'loss': 1.}, 0)
            with torch.no_grad():
                trainer.model.weight.fill_(2.)
            trainer.init_epoch = 4
            original = deepcopy(trainer.model.state_dict())

            def failure(preds, targets):
                raise RuntimeError('test metric failed')

            with self.assertRaisesRegex(RuntimeError, 'test metric failed'):
                self.quiet(trainer.test, self.loader(), metrics=[failure])
            self.assert_state_equal(trainer.model.state_dict(), original)
            self.assertEqual(trainer.init_epoch, 4)
            self.assertFalse(hasattr(trainer, '_test_model_state'))

    def test_resume_without_improvement_keeps_original_best_file(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            first = self.trainer([callback], running_id='first')
            callback.do_check(first, {'loss': 1.}, 0)
            original_path = callback.ckpt_path
            restored_callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            restored = self.trainer([restored_callback], running_id='continued')
            load_state(restored, original_path)
            restored_callback.do_check(restored, {'loss': 2.}, 1)
            self.assertEqual(restored_callback.ckpt_path, original_path)
            self.assertTrue(Path(original_path).is_file())
            self.assertFalse((Path(directory) / 'continued' / 'checkpoint.ckpt').exists())
            self.quiet(restored_callback.on_before_test_epochs, restored, [])
            restore_test_model(restored)

    def test_latest_ignores_newer_log_only_run(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([callback], running_id='with-checkpoint')
            callback.do_check(trainer, {'loss': 1.}, 0)
            log_only = Path(directory) / 'newer-logs'
            log_only.mkdir()
            (log_only / 'events.out.tfevents').write_text('log')
            os.utime(log_only, (2_000_000_000, 2_000_000_000))
            self.assertEqual(get_latest_running(directory), 'with-checkpoint')
            resumed_callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            resumed = self.trainer([resumed_callback], resume=True)
            self.quiet(resumed_callback.on_before_fit, resumed, 2)
            restore_fit_state(resumed, [], [])
            self.assertEqual(resumed.init_epoch, 1)
            self.assertEqual(resumed_callback.ckpt_path, callback.ckpt_path)

    def test_resume_true_allows_first_run_but_explicit_missing_run_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', ckpt_dir=directory)
            trainer = self.trainer([callback], resume=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                callback.on_before_fit(trainer, 1)
            self.assertIn('No checkpoint found', output.getvalue())
            self.assertEqual(trainer.init_epoch, 0)
            other_callback = CheckCallback('loss', ckpt_dir=directory)
            other = self.trainer([other_callback], resume='missing')
            with self.assertRaises(FileNotFoundError):
                self.quiet(other_callback.on_before_fit, other, 1)

    def test_existing_invalid_checkpoint_does_not_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid'
            path.mkdir()
            torch.save({'epoch': 2}, path / 'checkpoint.ckpt')
            callback = CheckCallback('loss', ckpt_dir=directory)
            trainer = self.trainer([callback], resume=True)
            with self.assertRaisesRegex(KeyError, 'checkpoint_file'):
                self.quiet(callback.on_before_fit, trainer, 1)
                restore_fit_state(trainer, [], [])
            self.assertEqual(trainer.init_epoch, 0)

    def test_multi_monitor_files_include_complete_current_callback_states(self):
        with tempfile.TemporaryDirectory() as directory:
            loss = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            score = CheckCallback('score', on_stage='train', mode='max', ckpt_dir=directory)
            trainer = self.trainer([loss, score], running_id='multiple')
            loss.do_check(trainer, {'loss': 2.}, 0, defer=True)
            score.do_check(trainer, {'score': 0.8}, 0, defer=True)
            self.assertIsNone(loss.ckpt_path)
            self.assertIsNone(score.ckpt_path)
            finalize_checkpoints(trainer)
            self.assertNotEqual(loss.ckpt_path, score.ckpt_path)
            for callback in (loss, score):
                state = torch.load(callback.ckpt_path, weights_only=False)
                values = {item['monitor']: item['best_value'] for item in state['callback_states']}
                self.assertEqual(values, {'loss': 2., 'score': 0.8})
                self.assertEqual({item['ordinal'] for item in state['callback_states']}, {0, 1})
                self.assertTrue(all(item['checkpoint_path'] for item in state['callback_states']))

    def test_fit_finalizes_all_epoch_monitors_after_their_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            loss_callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            score_callback = CheckCallback('score', on_stage='train', mode='max', ckpt_dir=directory)

            def score(preds, targets):
                return preds.mean()

            trainer = self.trainer([loss_callback, score_callback], metrics=[score])
            progress = self.quiet(trainer.fit, self.loader(), epochs=1)
            expected = {name: float(values[0]) for name, values in progress['train'].items()}
            for callback in (loss_callback, score_callback):
                state = torch.load(callback.ckpt_path, weights_only=False)
                self.assertEqual({item['monitor']: item['best_value'] for item in state['callback_states']},
                                 expected)

    def test_failed_save_keeps_previous_best_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([callback], running_id='original')
            callback.do_check(trainer, {'loss': 2.}, 0)
            original = callback.ckpt_path
            trainer.running_id = 'continued'
            with patch('deepepochs.callbacks.check.save_state', side_effect=OSError('save failed')):
                with self.assertRaisesRegex(OSError, 'save failed'):
                    callback.do_check(trainer, {'loss': 1.}, 1)
            self.assertEqual(callback.ckpt_path, original)
            self.assertTrue(Path(original).is_file())
            self.assertEqual(trainer._pending_checkpoints, [])
            finalize_checkpoints(trainer)

    def test_all_monitor_states_update_before_early_stop_is_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            loss = CheckCallback('loss', on_stage='train', patience=1, save_best=False, ckpt_dir=directory)
            score = CheckCallback('score', on_stage='train', mode='max', ckpt_dir=directory)
            trainer = self.trainer([loss, score])
            loss.do_check(trainer, {'loss': 1.}, 0, defer=True)
            score.do_check(trainer, {'score': 0.5}, 0, defer=True)
            finalize_checkpoints(trainer)
            loss.do_check(trainer, {'loss': 2.}, 1, defer=True)
            score.do_check(trainer, {'score': 0.9}, 1, defer=True)
            with self.assertRaises(StopLoopException):
                finalize_checkpoints(trainer)
            for callback in (loss, score):
                state = torch.load(callback.ckpt_path, weights_only=False)
                self.assertEqual(state['epoch'], 1)
                states = {item['monitor']: item for item in state['callback_states']}
                self.assertEqual(states['loss']['worse_times'], 1)
                self.assertEqual(states['score']['best_value'], 0.9)

    def test_duplicate_monitor_callbacks_restore_separate_ordinal_states(self):
        with tempfile.TemporaryDirectory() as directory:
            callbacks = [CheckCallback('loss', on_stage='train', ckpt_dir=directory) for _ in range(2)]
            trainer = self.trainer(callbacks)
            callbacks[0].do_check(trainer, {'loss': 1.}, 0, defer=True)
            callbacks[1].do_check(trainer, {'loss': 2.}, 0, defer=True)
            finalize_checkpoints(trainer)
            restored_callbacks = [CheckCallback('loss', on_stage='train', ckpt_dir=directory) for _ in range(2)]
            restored = self.trainer(restored_callbacks)
            load_state(restored, callbacks[0].ckpt_path)
            self.assertEqual([callback.best_value for callback in restored_callbacks], [1., 2.])


if __name__ == '__main__':
    unittest.main()
