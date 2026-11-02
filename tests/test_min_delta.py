"""严格最佳权重与累计足够改善的早停基准。"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from deepepochs import CheckCallback, Trainer
from deepepochs.callbacks.check import load_state
from deepepochs.loops import StopLoopException


class MinDeltaTests(unittest.TestCase):
    def trainer(self, callback):
        model = nn.Linear(1, 1, bias=False)
        trainer = Trainer(model, nn.functional.mse_loss,
                          torch.optim.SGD(model.parameters(), lr=0.1), callbacks=[callback],
                          device='cpu', show_info=False, log_batch=False, running_id='run')
        self.addCleanup(trainer.close)
        return trainer

    def test_micro_improvements_update_best_weights_and_accumulate_against_baseline(self):
        for mode, values in [('min', [10., 9.75, 9.5, 9.]), ('max', [10., 10.25, 10.5, 11.])]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                callback = CheckCallback('score', mode=mode, min_delta=0.5, patience=3, ckpt_dir=directory)
                trainer = self.trainer(callback)
                for epoch, value in enumerate(values):
                    with torch.no_grad():
                        trainer.model.weight.fill_(epoch + 1.)
                    callback.do_check(trainer, {'score': value}, epoch)
                    self.assertEqual(callback.best_value, value)
                    state = torch.load(callback.ckpt_path, weights_only=False)
                    self.assertEqual(float(state['model_state']['weight']), epoch + 1.)
                    self.assertEqual(state['callback_states'][0]['early_stop_baseline'], callback.early_stop_baseline)
                    self.assertEqual(callback.worse_times, [0, 1, 2, 0][epoch])
                self.assertEqual(callback.early_stop_baseline, values[-1])

    def test_resume_keeps_sufficient_improvement_baseline_and_prior_best_path(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', min_delta=0.5, patience=3, ckpt_dir=directory)
            trainer = self.trainer(callback)
            callback.do_check(trainer, {'loss': 10.}, 0)
            callback.do_check(trainer, {'loss': 9.75}, 1)
            source = callback.ckpt_path
            restored_callback = CheckCallback('loss', min_delta=0.5, patience=3, ckpt_dir=directory)
            restored = self.trainer(restored_callback)
            restored.running_id = 'continued'
            load_state(restored, source)
            self.assertEqual((restored_callback.best_value, restored_callback.early_stop_baseline,
                              restored_callback.worse_times), (9.75, 10., 1))
            restored_callback.do_check(restored, {'loss': 9.8}, 2)
            self.assertEqual(restored_callback.ckpt_path, source)
            restored_callback.do_check(restored, {'loss': 9.4}, 3)
            self.assertEqual((restored_callback.early_stop_baseline, restored_callback.worse_times), (9.4, 0))
            self.assertEqual(Path(restored_callback.ckpt_path).parent.name, 'continued')

    def test_patience_counts_insufficient_strict_improvements(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', min_delta=1., patience=2, ckpt_dir=directory)
            trainer = self.trainer(callback)
            callback.do_check(trainer, {'loss': 10.}, 0)
            callback.do_check(trainer, {'loss': 9.8}, 1)
            with self.assertRaises(StopLoopException):
                callback.do_check(trainer, {'loss': 9.6}, 2)
            self.assertEqual(callback.best_value, 9.6)
            self.assertEqual(callback.early_stop_baseline, 10.)
            self.assertEqual(torch.load(callback.ckpt_path, weights_only=False)['epoch'], 2)

    def test_zero_delta_uses_strict_improvement_and_rejects_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', patience=2, ckpt_dir=directory)
            trainer = self.trainer(callback)
            for epoch, value in enumerate([1., 0.99, 0.99]):
                callback.do_check(trainer, {'loss': value}, epoch)
            self.assertEqual((callback.best_value, callback.early_stop_baseline, callback.worse_times), (0.99, 0.99, 1))
        with self.assertRaises(ValueError):
            CheckCallback('loss', min_delta=-0.1)


if __name__ == '__main__':
    unittest.main()
