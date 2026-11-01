"""Checkpoint恢复优化器、调度器及监控状态，重现下一步更新。"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import CheckCallback, Optimizer, Trainer
from deepepochs.callbacks.check import load_state, save_state


class CheckpointTests(unittest.TestCase):
    def make_trainer(self, callbacks=None):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(.3)
        raw = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(raw, 1, gamma=.5)
        trainer = Trainer(model, nn.functional.mse_loss, Optimizer(raw, scheduler), epochs=1,
                          device='cpu', callbacks=callbacks, show_info=False, log_batch=False)
        return trainer, scheduler

    def fit(self, trainer):
        dl = DataLoader(TensorDataset(torch.tensor([[1.], [2.]]), torch.tensor([[2.], [4.]])), batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            return trainer.fit(dl, epochs=1)

    def test_resume_restores_optimizer_scheduler_and_next_update(self):
        trainer, scheduler = self.make_trainer()
        self.fit(trainer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.ckpt'
            save_state(trainer, trainer.model, trainer.opt, path, epoch=0)
            restored, restored_scheduler = self.make_trainer()
            load_state(restored, path)
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual(restored.init_epoch, 1)
            self.fit(trainer)
            dl = DataLoader(TensorDataset(torch.tensor([[1.], [2.]]), torch.tensor([[2.], [4.]])), batch_size=2)
            with contextlib.redirect_stdout(io.StringIO()):
                restored.fit(dl, epochs=2)
            torch.testing.assert_close(restored.model.weight, trainer.model.weight)
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_monitor_state_restores_and_rejects_worse_result(self):
        with tempfile.TemporaryDirectory() as directory:
            cb = CheckCallback('loss', on_stage='train', patience=3, save_best=False, ckpt_dir=directory)
            trainer, _ = self.make_trainer([cb])
            cb.do_check(trainer, {'loss': torch.tensor(1.)}, 0)
            cb.do_check(trainer, {'loss': torch.tensor(2.)}, 1)
            new_cb = CheckCallback('loss', on_stage='train', patience=3, ckpt_dir=directory)
            restored, _ = self.make_trainer([new_cb])
            load_state(restored, cb.ckpt_path)
            self.assertEqual((new_cb.best_value, new_cb.worse_times), (1., 1))
            new_cb.do_check(restored, {'loss': torch.tensor(3.)}, 2)
            self.assertEqual((new_cb.best_value, new_cb.worse_times), (1., 2))


if __name__ == '__main__':
    unittest.main()
