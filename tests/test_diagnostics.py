"""接口错误、监控配置、Task属性访问及正常早停。"""
import contextlib
import io
from pathlib import Path
import tempfile
import traceback
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, EpochTask, Trainer
from deepepochs.callbacks import CallbackException
from deepepochs.callbacks.check import save_state
from deepepochs.loops import LoopException


class FitFinished(Callback):
    def __init__(self):
        super().__init__()
        self.finished = False

    def on_after_fit(self, trainer):
        self.finished = True


class InvalidResult(EpochTask):
    def step(self, batch_x, batch_y, result):
        self.loss(self.model(batch_x), batch_y)
        return result


class DiagnosticsTests(unittest.TestCase):
    def loader(self):
        return DataLoader(TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)), batch_size=2)

    def trainer(self, callbacks=None, trainer_type=Trainer, **kwargs):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        trainer = trainer_type(model, nn.functional.mse_loss,
                               torch.optim.SGD(model.parameters(), lr=0.),
                               device='cpu', epochs=3, show_info=False, log_batch=False,
                               callbacks=callbacks, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_invalid_step_results_raise_in_training_and_testing(self):
        for operation in ('fit', 'test'):
            for result in (1., {'mae': torch.tensor(1.)}):
                with self.subTest(operation=operation, result=result):
                    finished = FitFinished()
                    trainer = self.trainer([finished])
                    task = InvalidResult(self.loader(), result=result)
                    arguments = {'train_tasks': [task]} if operation == 'fit' else {'tasks': [task]}
                    with self.assertRaisesRegex(LoopException, '返回'):
                        self.quiet(getattr(trainer, operation), **arguments)
                    self.assertFalse(finished.finished)
                    self.assertEqual(trainer.init_epoch, 0)

    def test_missing_monitor_raises_without_reporting_fit_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            finished = FitFinished()
            checker = CheckCallback('losss', on_stage='train', ckpt_dir=directory)
            trainer = self.trainer([finished, checker])
            with self.assertRaisesRegex(CallbackException, 'losss'):
                self.quiet(trainer.fit, self.loader())
            self.assertEqual(trainer.init_epoch, 1)
            self.assertFalse(finished.finished)

    def test_callback_error_keeps_its_exception_and_user_frame(self):
        error = CallbackException('custom callback failed')

        class FailBatch(Callback):
            def on_after_train_batch(self, trainer, metrics, batch_idx):
                raise error

        trainer = self.trainer([FailBatch()])
        try:
            self.quiet(trainer.fit, self.loader())
        except CallbackException as caught:
            self.assertIs(caught, error)
            frames = traceback.extract_tb(caught.__traceback__)
            self.assertEqual(frames[-1].name, 'on_after_train_batch')
        else:
            self.fail('CallbackException must reach the caller')

    def test_missing_task_attribute_identifies_the_misspelling(self):
        class MisspelledModel(EpochTask):
            def train_step(self, batch_x, batch_y):
                self.modle(batch_x)

        task = MisspelledModel(self.loader())
        self.assertFalse(hasattr(task, 'modle'))
        trainer = self.trainer()
        with self.assertRaisesRegex(AttributeError, 'modle'):
            self.quiet(trainer.fit, train_tasks=[task])
        self.assertFalse(hasattr(task, 'modle'))
        self.assertIs(task.model, trainer.model)

    def test_optional_stage_methods_keep_the_documented_selection_order(self):
        stages = []

        class CommonStepTrainer(Trainer):
            def step(self, batch_x, batch_y):
                stages.append(self.model.stage)
                self.loss(self.model(batch_x), batch_y)

        class DatasetTask(EpochTask):
            def train_step(self, batch_x, batch_y):
                raise AssertionError('Common step has priority over stage-specific methods')

        trainer = self.trainer(trainer_type=CommonStepTrainer)
        task = DatasetTask(self.loader(), name='dataset')
        history = self.quiet(trainer.fit, train_tasks=[task], val_tasks=[task], epochs=1)
        result = self.quiet(trainer.test, tasks=[task])
        self.assertEqual(stages, ['train', 'train', 'val', 'val', 'test', 'test'])
        self.assertEqual(set(history['train']), {'dataset/loss'})
        self.assertEqual(set(result), {'dataset/loss'})

    def test_evaluation_contract_error_restores_selected_weights(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.ckpt'
            with torch.no_grad():
                trainer.model.model.weight.fill_(1.)
            save_state(trainer, trainer.model, trainer.opt, str(path), epoch=0)
            with torch.no_grad():
                trainer.model.model.weight.fill_(2.)
            with self.assertRaises(LoopException):
                self.quiet(trainer.test, tasks=[InvalidResult(self.loader(), result=1.)], weights=path)
            self.assertEqual(float(trainer.model.model.weight.detach()), 2.)
            self.assertFalse(hasattr(trainer, '_test_model_state'))
            self.assertEqual(trainer.init_epoch, 0)

    def test_early_stopping_still_returns_history_and_finishes_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = CheckCallback('loss', on_stage='train', patience=1, ckpt_dir=directory)
            finished = FitFinished()
            trainer = self.trainer([checker, finished])
            history = self.quiet(trainer.fit, self.loader())
            self.assertEqual(len(history['train']['loss']), 2)
            self.assertEqual(trainer.init_epoch, 2)
            self.assertTrue(finished.finished)


if __name__ == '__main__':
    unittest.main()
