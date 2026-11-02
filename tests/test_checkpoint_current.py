"""当前Checkpoint的搬迁及中途退出收尾。"""
import contextlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, Trainer
from deepepochs.callbacks.check import finalize_checkpoints, load_state, restore_fit_state, _broadcast_state
from deepepochs.loops import StopLoopException


class CheckpointCurrentTests(unittest.TestCase):
    def trainer(self, callbacks=None, running_id='run', resume=False):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.3)
        trainer = Trainer(model, nn.functional.mse_loss,
                          torch.optim.SGD(model.parameters(), lr=0.01),
                          callbacks=callbacks, running_id=running_id, resume=resume,
                          device='cpu', epochs=1, show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        return trainer

    def loader(self):
        return DataLoader(TensorDataset(torch.tensor([[1.], [2.]]), torch.tensor([[2.], [4.]])),
                          batch_size=2)

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def separate_directory_snapshots(self, directory):
        first = CheckCallback('loss', on_stage='train', ckpt_dir=str(Path(directory) / 'a'))
        second = CheckCallback('score', on_stage='train', mode='max', ckpt_dir=str(Path(directory) / 'b'))
        trainer = self.trainer([first, second], running_id='saved')
        with torch.no_grad():
            trainer.model.weight.fill_(1.)
        first.do_check(trainer, {'loss': 1.}, 0, defer=True)
        second.do_check(trainer, {'score': 1.}, 0, defer=True)
        finalize_checkpoints(trainer)
        with torch.no_grad():
            trainer.model.weight.fill_(2.)
        first.do_check(trainer, {'loss': 0.5}, 1, defer=True)
        second.do_check(trainer, {'score': 0.5}, 1, defer=True)
        finalize_checkpoints(trainer)
        os.utime(first.ckpt_path, (1_800_000_000, 1_800_000_000))
        os.utime(second.ckpt_path, (1_700_000_000, 1_700_000_000))
        return trainer, first, second

    def test_auto_resume_selects_latest_snapshot_across_monitor_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            source, first, second = self.separate_directory_snapshots(directory)
            for first_to_run in (0, 1):
                with self.subTest(first_to_run=first_to_run):
                    callbacks = [CheckCallback('loss', on_stage='train', ckpt_dir=first.ckpt_dir),
                                 CheckCallback('score', on_stage='train', mode='max', ckpt_dir=second.ckpt_dir)]
                    for index, callback in enumerate(callbacks):
                        callback.priority = -2 if index == first_to_run else -1
                    restored = self.trainer(callbacks, resume=True)
                    with patch('deepepochs.callbacks.check._broadcast_state', wraps=_broadcast_state) as load:
                        self.quiet(restored.callbacks.trigger, 'before_fit', trainer=restored, epochs=3)
                        restore_fit_state(restored, [], [])
                    self.assertEqual(load.call_count, 1)
                    self.assertEqual(Path(load.call_args.args[1]), Path(first.ckpt_path))
                    self.assertEqual(restored.init_epoch, 2)
                    self.assertEqual(float(restored.model.weight.detach()), 2.)
                    self.assertEqual([callback.best_value for callback in callbacks], [0.5, 1.])
                    self.assertEqual([callback.worse_times for callback in callbacks], [0, 1])
                    self.assertTrue(all(callback._resume_loaded for callback in callbacks))

    def test_explicit_run_selects_latest_across_directories_and_excludes_other_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, first, second = self.separate_directory_snapshots(directory)
            selected_path = first.ckpt_path
            source.running_id = 'other'
            with torch.no_grad():
                source.model.weight.fill_(3.)
            first.do_check(source, {'loss': 0.25}, 2)
            os.utime(first.ckpt_path, (1_900_000_000, 1_900_000_000))
            callbacks = [CheckCallback('loss', on_stage='train', ckpt_dir=first.ckpt_dir),
                         CheckCallback('score', on_stage='train', mode='max', ckpt_dir=second.ckpt_dir)]
            callbacks[0].priority, callbacks[1].priority = -1, -2
            restored = self.trainer(callbacks, resume='saved')
            with patch('deepepochs.callbacks.check._broadcast_state', wraps=_broadcast_state) as load:
                self.quiet(restored.callbacks.trigger, 'before_fit', trainer=restored, epochs=3)
                restore_fit_state(restored, [], [])
            self.assertEqual(load.call_count, 1)
            self.assertEqual(Path(load.call_args.args[1]), Path(selected_path))
            self.assertEqual(restored.init_epoch, 2)
            self.assertEqual(float(restored.model.weight.detach()), 2.)

    def test_auto_resume_without_any_monitor_snapshot_starts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            callbacks = [CheckCallback('loss', ckpt_dir=str(Path(directory) / name)) for name in ('a', 'b')]
            trainer = self.trainer(callbacks, resume=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                trainer.callbacks.trigger('before_fit', trainer=trainer, epochs=1)
            self.assertEqual(output.getvalue().count('No checkpoint found'), 1)
            self.assertEqual(trainer.init_epoch, 0)
            self.assertAlmostEqual(float(trainer.model.weight.detach()), 0.3, places=6)
            self.assertTrue(all(callback._resume_loaded for callback in callbacks))

    def test_single_checkpoint_can_be_moved_and_renamed_before_loading_and_testing(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([callback])
            callback.do_check(trainer, {'loss': 1.}, 0)
            original = Path(callback.ckpt_path)
            destination = Path(directory) / 'moved' / 'renamed.ckpt'
            destination.parent.mkdir()
            shutil.move(original, destination)
            self.assertFalse(original.exists())
            restored_callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            restored = self.trainer([restored_callback])
            load_state(restored, destination)
            self.assertEqual(restored_callback.ckpt_path, str(destination))
            with torch.no_grad():
                restored.model.weight.fill_(0.9)
            seen = []

            def metric(preds, targets):
                seen.append(float(restored.model.weight.detach()))
                return preds.mean()

            self.quiet(restored.test, self.loader(), metrics=[metric])
            self.assertTrue(seen)
            self.assertTrue(all(abs(value - 0.3) < 1e-6 for value in seen))
            self.assertAlmostEqual(float(restored.model.weight.detach()), 0.9, places=6)

    def create_different_best_files(self, directory, cross_run=False):
        first = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
        second = CheckCallback('score', on_stage='train', mode='max', ckpt_dir=directory)
        trainer = self.trainer([first, second], running_id='original')
        first.do_check(trainer, {'loss': 1.}, 0, defer=True)
        second.do_check(trainer, {'score': 1.}, 0, defer=True)
        finalize_checkpoints(trainer)
        with torch.no_grad():
            trainer.model.weight.fill_(0.9)
        if cross_run:
            trainer.running_id = 'continued'
            first.do_check(trainer, {'loss': 0.5}, 1, defer=True)
            second.do_check(trainer, {'score': 0.5}, 1, defer=True)
            source = first
        else:
            first.do_check(trainer, {'loss': 2.}, 1, defer=True)
            second.do_check(trainer, {'score': 2.}, 1, defer=True)
            source = second
        finalize_checkpoints(trainer)
        return first, second, Path(source.ckpt_path)

    def test_moved_multi_monitor_directory_keeps_distinct_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second, snapshot = self.create_different_best_files(directory)
            saved = torch.load(snapshot, weights_only=False)
            self.assertEqual(saved['checkpoint_file'], snapshot.name)
            self.assertEqual([item['checkpoint_path'] for item in saved['callback_states']],
                             ['checkpoint-1.ckpt', 'checkpoint-2.ckpt'])
            moved = Path(directory) / 'relocated'
            shutil.move(snapshot.parent, moved)
            restored_callbacks = [CheckCallback('loss', on_stage='train'),
                                  CheckCallback('score', on_stage='train', mode='max')]
            restored = self.trainer(restored_callbacks)
            load_state(restored, moved / snapshot.name)
            expected_weights = (0.3, 0.9)
            for callback, expected in zip(restored_callbacks, expected_weights):
                self.assertEqual(Path(callback.ckpt_path).parent, moved)
                weight = torch.load(callback.ckpt_path, weights_only=False)['model_state']['weight']
                self.assertAlmostEqual(float(weight), expected, places=6)
            self.assertNotEqual(restored_callbacks[0].ckpt_path, restored_callbacks[1].ckpt_path)

    def test_moved_checkpoint_tree_preserves_best_file_from_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'tree'
            first, second, snapshot = self.create_different_best_files(original, cross_run=True)
            state = torch.load(snapshot, weights_only=False)
            self.assertEqual(state['callback_states'][1]['checkpoint_path'], '../original/checkpoint-2.ckpt')
            moved = Path(directory) / 'moved-tree'
            shutil.move(original, moved)
            restored_callbacks = [CheckCallback('loss', on_stage='train'),
                                  CheckCallback('score', on_stage='train', mode='max')]
            restored = self.trainer(restored_callbacks)
            load_state(restored, moved / 'continued' / snapshot.name)
            self.assertEqual(Path(restored_callbacks[0].ckpt_path), moved / 'continued' / 'checkpoint-1.ckpt')
            self.assertEqual(Path(restored_callbacks[1].ckpt_path), moved / 'original' / 'checkpoint-2.ckpt')
            for callback, expected in zip(restored_callbacks, (0.9, 0.3)):
                weight = torch.load(callback.ckpt_path, weights_only=False)['model_state']['weight']
                self.assertAlmostEqual(float(weight), expected, places=6)

    def test_custom_stop_flushes_pending_without_replacing_reason_or_polluting_next_fit(self):
        class StopBetweenMonitors(Callback):
            def __init__(self):
                super().__init__()
                self.priority = 1
                self.enabled = True

            def on_after_epoch(self, *args, **kwargs):
                if self.enabled:
                    raise StopLoopException('custom stopping reason')

        with tempfile.TemporaryDirectory() as directory:
            first = CheckCallback('loss', on_stage='train', save_best=False, patience=1, ckpt_dir=directory)
            second = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=directory)
            first.priority, second.priority = 2, -1
            first.best_value = 0.
            stopper = StopBetweenMonitors()
            trainer = self.trainer([first, stopper, second])
            epochs_saved = []
            save = torch.save

            def record(state, path, **kwargs):
                epochs_saved.append(state['epoch'])
                return save(state, path, **kwargs)

            output = io.StringIO()
            with patch('deepepochs.callbacks.check.torch.save', side_effect=record):
                with contextlib.redirect_stdout(output):
                    trainer.fit(self.loader())
                self.assertEqual(epochs_saved, [0])
                self.assertEqual(trainer._pending_checkpoints, [])
                self.assertTrue(Path(first.ckpt_path).is_file())
                self.assertIsNone(second.ckpt_path)
                stopper.enabled = False
                first.patience = 0
                self.quiet(trainer.fit, self.loader(), epochs=1)
            self.assertIn('custom stopping reason', output.getvalue())
            self.assertNotIn('Early stopping triggered', output.getvalue())
            self.assertEqual(epochs_saved, [0, 1, 1])
            self.assertEqual(trainer._pending_checkpoints, [])

    def test_early_stopping_works_inside_callers_exception_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('fixed', on_stage='train', patience=1, ckpt_dir=directory)
            trainer = self.trainer([callback])
            trainer.max_epochs = 5

            def fixed(preds, targets):
                return 1.

            output = io.StringIO()
            try:
                raise RuntimeError('previous attempt failed')
            except RuntimeError:
                with contextlib.redirect_stdout(output):
                    progress = trainer.fit(self.loader(), metrics=[fixed])
            self.assertEqual(len(progress['train']['loss']), 2)
            self.assertEqual(trainer.init_epoch, 2)
            self.assertIn('Early stopping triggered', output.getvalue())
            self.assertEqual(trainer._pending_checkpoints, [])

    def test_save_failure_chains_current_callback_error(self):
        class ErrorAfterMonitor(Callback):
            def __init__(self):
                super().__init__()
                self.priority = 1

            def on_after_epoch(self, *args, **kwargs):
                raise RuntimeError('after_epoch failed')

        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            callback.priority = 2
            trainer = self.trainer([callback, ErrorAfterMonitor()])
            with patch('deepepochs.callbacks.check.save_state', side_effect=OSError('save failed')):
                with self.assertRaisesRegex(OSError, 'save failed') as caught:
                    self.quiet(trainer.fit, self.loader())
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertEqual(str(caught.exception.__cause__), 'after_epoch failed')
            self.assertEqual(trainer._pending_checkpoints, [])

    def test_callback_error_flushes_pending_and_propagates_original_error(self):
        class ErrorAfterMonitor(Callback):
            def __init__(self):
                super().__init__()
                self.priority = 1

            def on_after_epoch(self, *args, **kwargs):
                raise RuntimeError('after_epoch failed')

        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            callback.priority = 2
            trainer = self.trainer([callback, ErrorAfterMonitor()])
            with self.assertRaisesRegex(RuntimeError, 'after_epoch failed'):
                self.quiet(trainer.fit, self.loader())
            self.assertEqual(trainer._pending_checkpoints, [])
            self.assertTrue(Path(callback.ckpt_path).is_file())


if __name__ == '__main__':
    unittest.main()
