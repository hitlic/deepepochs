"""检查点与评估提示反映已经完成的操作，并保留训练状态。"""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, Trainer
from deepepochs.callbacks.check import finalize_checkpoints
from deepepochs.loops import StopLoopException


class ConsoleEventTests(unittest.TestCase):
    def trainer(self, callbacks=None):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.25)
        trainer = Trainer(model, nn.functional.mse_loss,
                          torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
                          callbacks=callbacks, device='cpu', show_info=False,
                          log_batch=False, running_id='console')
        self.addCleanup(trainer.close)
        return trainer

    def loader(self):
        return DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=2)

    def capture(self, function, *args, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = function(*args, **kwargs)
        return result, output.getvalue()

    def test_successful_saves_are_silent_and_final_summaries_keep_saved_values(self):
        with tempfile.TemporaryDirectory() as directory:
            best = CheckCallback('main/loss', on_stage='train', ckpt_dir=directory)
            latest = CheckCallback('main/score', on_stage='val', mode='max',
                                   save_best=False, ckpt_dir=directory)
            trainer = self.trainer([best, latest])
            best.do_check(trainer, {'main/loss': 2.}, 0, defer=True)
            latest.do_check(trainer, {'main/score': 0.9}, 0, defer=True)
            _, output = self.capture(finalize_checkpoints, trainer)
            self.assertEqual(output, '')
            latest.do_check(trainer, {'main/score': 0.1}, 1, defer=True)
            _, output = self.capture(finalize_checkpoints, trainer)
            self.assertEqual(output, '')
            latest.save_best = True
            events = []

            def observe(message):
                selected = best if message.startswith('[BEST]') else latest
                self.assertIn(str(selected.ckpt_path), message)
                self.assertTrue(Path(selected.ckpt_path).is_file())
                state = torch.load(selected.ckpt_path, map_location='cpu', weights_only=False)
                self.assertEqual(state['epoch'], 0 if selected is best else 1)
                events.append(message)

            with patch.object(trainer, 'print', side_effect=observe):
                best.on_after_fit(trainer)
                latest.on_after_fit(trainer)
            self.assertEqual(len(events), 2)
            self.assertIn('[BEST] train main/loss: 2.0000', events[0])
            self.assertIn('epoch=1', events[0])
            self.assertIn('[LATEST] val main/score: 0.1000', events[1])
            self.assertIn('epoch=2', events[1])
            self.assertEqual(latest.best_value, 0.9)

    def test_direct_check_defers_success_message_until_fit_ends(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([checker])
            keep, output = self.capture(checker.check, trainer, {'loss': 0.5},
                                        trainer.model, trainer.opt, 2)
            self.assertTrue(keep)
            self.assertEqual(output, '')
            self.assertTrue(Path(checker.ckpt_path).is_file())
            _, output = self.capture(checker.on_after_fit, trainer)
            self.assertIn('[BEST] train loss: 0.5000', output)
            self.assertIn('epoch=3', output)

    def test_failed_save_keeps_path_and_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([checker])
            self.capture(checker.do_check, trainer, {'loss': 2.}, 0)
            original = checker.ckpt_path
            checker.on_before_fit(trainer, epochs=2)
            trainer.running_id = 'failed'
            failure = OSError('disk write failed')
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch('deepepochs.callbacks.check.save_state', side_effect=failure):
                with self.assertRaises(OSError) as raised:
                    checker.do_check(trainer, {'loss': 1.}, 1)
            self.assertIs(raised.exception, failure)
            self.assertNotIn('[BEST]', output.getvalue())
            self.assertNotIn('[LATEST]', output.getvalue())
            self.assertEqual(checker.ckpt_path, original)
            self.assertEqual(trainer._pending_checkpoints, [])
            _, output = self.capture(checker.on_after_fit, trainer)
            self.assertIn('[CHECKPOINT]', output)
            self.assertNotIn('[BEST]', output)
            self.assertIn(str(original), output)

    def test_early_stopping_reason_contains_best_and_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', patience=1,
                                    save_best=False, ckpt_dir=directory)
            trainer = self.trainer([checker])
            self.capture(checker.do_check, trainer, {'loss': 1.}, 0)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(StopLoopException) as raised:
                checker.do_check(trainer, {'loss': 2.}, 1)
            message = str(raised.exception)
            self.assertIn('Early stopping triggered: train loss', message)
            self.assertIn('best=1', message)
            self.assertIn('patience=1/1', message)
            self.assertEqual((checker.best_value, checker.worse_times), (1., 1))
            self.assertEqual(torch.load(checker.ckpt_path, weights_only=False)['epoch'], 1)
            _, output = self.capture(checker.on_after_fit, trainer)
            self.assertIn('[LATEST] train loss: 2.0000', output)
            self.assertIn('epoch=2', output)

    def test_fit_reports_each_monitor_once_after_the_last_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            best = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            latest = CheckCallback('loss', on_stage='val', save_best=False, ckpt_dir=directory)
            trainer = self.trainer([best, latest])
            history, output = self.capture(trainer.fit, self.loader(), self.loader(), epochs=3)
            lines = output.splitlines()
            summaries = [index for index, line in enumerate(lines) if line.startswith('E ')]
            self.assertEqual(len(summaries), 3)
            for label, checker, stage in (('BEST', best, 'train'), ('LATEST', latest, 'val')):
                events = [(index, line) for index, line in enumerate(lines) if f'[{label}]' in line]
                self.assertEqual(len(events), 1)
                index, event = events[0]
                self.assertGreater(index, summaries[-1])
                self.assertIn(f'{stage} loss: {history[stage]["loss"][-1]:.4f}', event)
                self.assertIn('epoch=3', event)
                self.assertEqual(torch.load(checker.ckpt_path, weights_only=False)['epoch'], 2)

    def test_early_stopped_fit_reports_one_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', save_best=False,
                                    patience=1, ckpt_dir=directory)
            trainer = self.trainer([checker])
            for optimizer in trainer.opt:
                for group in optimizer.param_groups:
                    group['lr'] = 0
            _, output = self.capture(trainer.fit, self.loader(), epochs=5)
            self.assertEqual(trainer.init_epoch, 2)
            self.assertEqual(output.count('[LATEST]'), 1)
            event = next(line for line in output.splitlines() if '[LATEST]' in line)
            self.assertIn('epoch=2', event)
            self.assertGreater(output.index('[LATEST]'), output.index('Early stopping triggered'))
            self.assertEqual(torch.load(checker.ckpt_path, weights_only=False)['epoch'], 1)

    def test_keyboard_interrupt_reports_the_last_completed_checkpoint(self):
        class Interrupt(Callback):
            def on_before_train_batch(self, trainer, **kwargs):
                if trainer.epoch_idx == 1:
                    raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=directory)
            trainer = self.trainer([checker, Interrupt()])
            _, output = self.capture(trainer.fit, self.loader(), epochs=3)
            self.assertEqual(trainer.init_epoch, 1)
            self.assertEqual(output.count('[LATEST]'), 1)
            event = next(line for line in output.splitlines() if '[LATEST]' in line)
            self.assertIn('epoch=1', event)
            self.assertGreater(output.index('[LATEST]'), output.index('Training interrupted by user'))
            self.assertEqual(torch.load(checker.ckpt_path, weights_only=False)['epoch'], 0)

    def test_repeated_fit_without_a_new_save_reports_only_the_existing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([checker])
            for optimizer in trainer.opt:
                for group in optimizer.param_groups:
                    group['lr'] = 0
            _, first = self.capture(trainer.fit, self.loader(), epochs=1)
            self.assertEqual(first.count('[BEST]'), 1)
            _, second = self.capture(trainer.fit, self.loader(), epochs=1)
            self.assertNotIn('[BEST]', second)
            self.assertNotIn('[LATEST]', second)
            events = [line for line in second.splitlines() if '[CHECKPOINT]' in line]
            self.assertEqual(len(events), 1)
            self.assertIn(f'monitor: train/loss | checkpoint: {checker.ckpt_path}', events[0])
            self.assertNotIn('epoch=', events[0])
            self.assertEqual(torch.load(checker.ckpt_path, weights_only=False)['epoch'], 0)

    def test_restored_checkpoint_without_a_new_save_has_a_neutral_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=directory)
            original = self.trainer([latest])
            for optimizer in original.opt:
                for group in optimizer.param_groups:
                    group['lr'] = 0
            self.capture(original.fit, self.loader(), epochs=1)
            checker = CheckCallback('loss', on_stage='train', save_best=True, ckpt_dir=directory)
            restored = self.trainer([checker])
            self.capture(restored.load_checkpoint, latest.ckpt_path)
            _, output = self.capture(restored.fit, self.loader(), epochs=2)
            self.assertEqual(restored.init_epoch, 2)
            self.assertNotIn('[BEST]', output)
            self.assertNotIn('[LATEST]', output)
            events = [line for line in output.splitlines() if '[CHECKPOINT]' in line]
            self.assertEqual(len(events), 1)
            self.assertIn(f'monitor: train/loss | checkpoint: {checker.ckpt_path}', events[0])
            self.assertNotIn('epoch=', events[0])
            self.assertEqual(torch.load(checker.ckpt_path, weights_only=False)['epoch'], 0)

    def test_evaluation_messages_identify_effective_weights_and_preserve_current_state(self):
        with tempfile.TemporaryDirectory() as directory:
            best = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            latest = CheckCallback('score', on_stage='train', mode='max',
                                   save_best=False, ckpt_dir=directory)
            trainer = self.trainer([best, latest])
            with torch.no_grad():
                trainer.model.weight.fill_(1.)
            self.capture(best.do_check, trainer, {'loss': 1.}, 0)
            with torch.no_grad():
                trainer.model.weight.fill_(2.)
            self.capture(latest.do_check, trainer, {'score': 0.5}, 1)
            with torch.no_grad():
                trainer.model.weight.fill_(3.)
            before_opt = deepcopy(trainer.opt.state_dict())
            choices = [('current', 9., None, None, None),
                       ('auto', 1., best.ckpt_path, 'train/loss', 1),
                       (latest, 4., latest.ckpt_path, 'train/score', 2),
                       (Path(best.ckpt_path), 1., best.ckpt_path, None, 1)]
            for choice, expected_loss, path, monitor, saved_epoch in choices:
                with self.subTest(weights=choice):
                    result, output = self.capture(trainer.test, self.loader(), weights=choice)
                    self.assertAlmostEqual(float(result['loss']), expected_loss, places=6)
                    if path is None:
                        self.assertIn('[WEIGHTS] Using current model weights', output)
                        self.assertNotIn('saved epoch:', output)
                    else:
                        self.assertIn(f'[WEIGHTS] Using checkpoint: {path}', output)
                        self.assertIn(f'saved epoch: {saved_epoch}', output)
                        if monitor is None:
                            self.assertNotIn('monitor:', output)
                        else:
                            self.assertIn(f'monitor: {monitor}', output)
                    self.assertEqual(float(trainer.model.weight.detach()), 3.)
                    self.assertEqual(trainer.opt.state_dict(), before_opt)
                    self.assertEqual(trainer.init_epoch, 0)

    def test_restored_latest_checkpoint_identifies_monitor_despite_new_save_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=directory)
            original = self.trainer([latest])
            for epoch, weight in enumerate((1., 2.)):
                with torch.no_grad():
                    original.model.weight.fill_(weight)
                self.capture(latest.do_check, original, {'loss': weight}, epoch)

            checker = CheckCallback('loss', on_stage='train', save_best=True, ckpt_dir=directory)
            restored = self.trainer([checker])
            self.capture(restored.load_checkpoint, latest.ckpt_path)
            self.assertTrue(checker.save_best)
            self.assertEqual(checker.best_value, 1.)
            self.assertEqual(float(restored.model.weight.detach()), 2.)
            with torch.no_grad():
                restored.model.weight.fill_(7.)
            before_opt = deepcopy(restored.opt.state_dict())

            result, test_output = self.capture(restored.test, self.loader(), weights=checker)
            self.assertEqual(float(result['loss']), 4.)
            self.assertEqual(float(restored.model.weight.detach()), 7.)
            inputs = DataLoader(torch.ones(2, 1), batch_size=1)
            predictions, predict_output = self.capture(list, restored.predict(inputs, weights='auto'))
            torch.testing.assert_close(torch.cat(predictions), torch.full((2, 1), 2.))
            for output in (test_output, predict_output):
                self.assertEqual(output.count('[WEIGHTS]'), 1)
                self.assertIn(f'[WEIGHTS] Using checkpoint: {checker.ckpt_path}', output)
                self.assertIn('monitor: train/loss', output)
                self.assertIn('saved epoch: 2', output)
            self.assertEqual(float(restored.model.weight.detach()), 7.)
            self.assertEqual(restored.opt.state_dict(), before_opt)
            self.assertEqual(restored.init_epoch, 2)

    def test_auto_without_checkpoint_reports_current_and_file_without_epoch_loads(self):
        trainer = self.trainer()
        result, output = self.capture(trainer.test, self.loader())
        self.assertIn('[WEIGHTS] Using current model weights', output)
        self.assertAlmostEqual(float(result['loss']), 0.25 ** 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.ckpt'
            torch.save({'model_state': {'weight': torch.ones(1, 1)}}, path)
            result, output = self.capture(trainer.test, self.loader(), weights=path)
            self.assertIn(f'[WEIGHTS] Using checkpoint: {path}', output)
            self.assertNotIn('saved epoch:', output)
            self.assertEqual(float(result['loss']), 1.)
            self.assertEqual(float(trainer.model.weight.detach()), 0.25)

    def test_non_main_rank_does_not_repeat_business_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([checker])
            trainer.main_process = False
            _, output = self.capture(checker.check, trainer, {'loss': 0.5},
                                      trainer.model, trainer.opt, 0)
            self.assertEqual(output, '')
            _, output = self.capture(checker.on_after_fit, trainer)
            self.assertEqual(output, '')
            self.assertFalse(Path(checker.ckpt_path).exists())
            _, output = self.capture(trainer.test, self.loader(), weights='current')
            self.assertEqual(output, '')


if __name__ == '__main__':
    unittest.main()
