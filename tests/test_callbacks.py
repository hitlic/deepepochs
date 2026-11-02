import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, EpochTask, Trainer
from deepepochs.callbacks import AnalyzeCallback, InterpreteCallback, LogCallback, LRFindCallback
from deepepochs.loops import StopLoopException
from deepepochs.metrics import confusion_matrix


class RecordingLogger:
    def __init__(self):
        self.figures = []
        self.images = []
        self.scalars = []
        self.histograms = []
        self.closed = False

    def add_figure(self, *args):
        self.figures.append(args)

    def add_image(self, *args):
        self.images.append(args)

    def add_scalar(self, *args, **kwargs):
        self.scalars.append(args)

    def add_histogram(self, *args, **kwargs):
        self.histograms.append(args)

    def close(self):
        self.closed = True


class CallbackTests(unittest.TestCase):
    def logging_trainer(self, callbacks):
        model = torch.nn.Linear(1, 2)
        trainer = Trainer(model, F.cross_entropy, torch.optim.SGD(model.parameters(), lr=.01),
                          epochs=1, callbacks=callbacks, device='cpu',
                          show_info=False, log_batch=False, running_id='logging')
        self.addCleanup(trainer.close)
        inputs = torch.tensor([[-1.], [1.], [-2.], [2.]])
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        return trainer, loader

    def test_training_loops_release_writer_threads_on_close(self):
        workers = []
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(3):
                callback = LogCallback(log_dir=directory)
                trainer, loader = self.logging_trainer([callback])
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.fit(loader)
                    trainer.test(loader)
                writer = trainer.logger
                # 保存本次writer的真实线程对象，close应同步等待线程退出。
                worker = writer.file_writer.event_writer._async_writer._worker
                workers.append(worker)
                try:
                    self.assertTrue(worker.is_alive())
                    with patch.object(writer, 'close', wraps=writer.close) as close:
                        trainer.close()
                        trainer.close()
                        close.assert_called_once()
                    self.assertFalse(worker.is_alive())
                    self.assertIsNone(getattr(trainer, 'logger', None))
                    self.assertIsNone(getattr(callback, 'logger', None))
                finally:
                    writer.close()
            self.assertFalse(any(worker.is_alive() for worker in workers))

    def test_callbacks_share_writer_through_fit_and_test_until_close(self):
        with tempfile.TemporaryDirectory() as directory:
            callbacks = [LogCallback(log_dir=directory),
                         InterpreteCallback(stages='test', log_dir=directory),
                         AnalyzeCallback(mode='forward', log_dir=directory)]
            trainer, loader = self.logging_trainer(callbacks)
            writer = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.fit(loader)
                writer = trainer.logger
                self.assertTrue(all(callback.logger is writer for callback in callbacks))
                # fit结束只移除分析hooks；随后test继续写入同一个event文件。
                self.assertIsNotNone(writer.file_writer)
                file_writer = writer.file_writer
                worker = file_writer.event_writer._async_writer._worker
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.test(loader)
                self.assertIs(writer.file_writer, file_writer)
                with patch.object(writer, 'close', wraps=writer.close) as close:
                    trainer.close()
                    trainer.close()
                    close.assert_called_once()
                self.assertFalse(worker.is_alive())
                self.assertTrue(all(getattr(callback, 'logger', None) is None for callback in callbacks))
            finally:
                if writer is not None:
                    writer.close()

    def test_test_only_interpreter_writer_is_released_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = InterpreteCallback(stages='test', log_dir=directory)
            trainer, loader = self.logging_trainer([callback])
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.test(loader)
            writer = trainer.logger
            worker = writer.file_writer.event_writer._async_writer._worker
            try:
                self.assertTrue(worker.is_alive())
                trainer.close()
                self.assertFalse(worker.is_alive())
                self.assertIsNone(getattr(trainer, 'logger', None))
                self.assertIsNone(getattr(callback, 'logger', None))
            finally:
                writer.close()

    def test_interpreter_infers_output_classes_and_preserves_shared_matrix(self):
        callback = InterpreteCallback()
        targets = torch.tensor([4, 5])
        predictions = torch.eye(6)[targets]
        shared = confusion_matrix(predictions, targets, 6)
        expected = shared.clone()
        callback.on_before_train_batch(None, torch.zeros(2, 3, 4, 4), targets, 0)
        callback.on_train_metrics(None, None, predictions, targets, None)
        callback.on_train_metrics(None, None, predictions, targets, None)
        self.assertEqual(callback.class_num, 6)
        torch.testing.assert_close(callback.confusion_matrix, expected * 2)
        torch.testing.assert_close(shared, expected)

    def test_explicit_interpreter_classes_take_precedence(self):
        callback = InterpreteCallback(class_num=8)
        callback.on_val_metrics(None, None, torch.eye(6)[[4]], torch.tensor([4]), None)
        self.assertEqual(callback.confusion_matrix.shape, (8, 8))

    def test_single_process_test_initializes_logger_and_selects_samples(self):
        logger = RecordingLogger()
        trainer = SimpleNamespace(main_process=True, running_id='test', logger=logger)
        with tempfile.TemporaryDirectory() as directory:
            callback = InterpreteCallback(metric=lambda preds, targets: preds[:, 1],
                                          k=1, stages='test', image_data=True, log_dir=directory)
            callback.on_before_test_epochs(trainer, [])
            inputs = torch.arange(8.).reshape(2, 1, 2, 2)
            targets = torch.tensor([1, 0])
            predictions = torch.tensor([[0.1, 0.9], [0.6, 0.4]])
            callback.on_before_test_batch(trainer, inputs, targets, 0)
            callback.on_test_metrics(trainer, None, predictions, targets, None)
            callback.on_after_test_epochs(trainer, [], {})
            self.assertEqual(callback.class_num, 2)
            self.assertAlmostEqual(callback.top()[0][0], 0.9, places=6)
            self.assertEqual(len(logger.figures), 1)
            self.assertEqual(len(logger.images), 1)
            callback.on_before_test_epochs(trainer, [])
            self.assertEqual(callback.top(), [])

    def test_empty_non_main_rank_gathers_without_logger(self):
        matrix = torch.tensor([[1, 0], [0, 1]])
        candidate = (0.9, [0.9, torch.tensor([0.1, 0.9]), torch.tensor(1), torch.ones(1, 2, 2)])

        class Context:
            is_distributed = True

            def __init__(self):
                self.calls = 0

            def all_gather_object(self, data):
                self.calls += 1
                if self.calls == 1:
                    self_test.assertIsNone(data)
                    return [matrix, None]
                self_test.assertEqual(data, [])
                return [[candidate], []]

        self_test = self
        context = Context()
        trainer = SimpleNamespace(main_process=False, distributed=context)
        callback = InterpreteCallback(stages='test', image_data=True)
        with patch('deepepochs.callbacks.interprete.SummaryWriter') as writer:
            callback.on_before_fit(trainer, 1)
            callback.on_before_test_epochs(trainer, [])
            callback.on_after_test_epochs(trainer, [], {})
            writer.assert_not_called()
        self.assertEqual(context.calls, 2)
        self.assertFalse(hasattr(callback, 'logger'))
        torch.testing.assert_close(callback.confusion_matrix, matrix)

    def test_main_rank_selects_global_top_candidates_even_when_locally_empty(self):
        matrix = torch.tensor([[0, 0], [0, 3]])
        candidates = [(score, [score, torch.tensor([0.1, 0.9]), torch.tensor(1), torch.ones(1, 2, 2)])
                      for score in (0.9, 0.7)]

        class Context:
            is_distributed = True

            def __init__(self):
                self.calls = 0

            def all_gather_object(self, data):
                self.calls += 1
                return [None, matrix] if self.calls == 1 else [[], candidates]

        logger = RecordingLogger()
        trainer = SimpleNamespace(main_process=True, running_id='test', logger=logger,
                                  distributed=Context())
        with tempfile.TemporaryDirectory() as directory:
            callback = InterpreteCallback(k=1, stages='test', image_data=True, log_dir=directory)
            callback.on_before_test_epochs(trainer, [])
            callback.on_after_test_epochs(trainer, [], {})
        self.assertEqual(callback.top()[0][0], 0.9)
        self.assertEqual(len(logger.images), 1)
        self.assertEqual(len(logger.figures), 1)
        self.assertEqual(callback.class_num, 2)

    def test_all_ranks_empty_interpreter_skips_plot(self):
        class Context:
            is_distributed = True

            def all_gather_object(self, data):
                return [data, data]

        logger = RecordingLogger()
        trainer = SimpleNamespace(main_process=True, running_id='test', logger=logger,
                                  distributed=Context())
        with tempfile.TemporaryDirectory() as directory:
            callback = InterpreteCallback(stages='test', log_dir=directory)
            callback.on_before_test_epochs(trainer, [])
            callback.on_after_test_epochs(trainer, [], {})
        self.assertEqual(logger.figures, [])

    def test_analyze_uses_all_samples_for_forward_and_backward_statistics(self):
        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.fill_(1.)
                logger = RecordingLogger()
                trainer = SimpleNamespace(main_process=True, running_id='test', logger=logger, model=model)
                callback = AnalyzeCallback(mode=mode, log_dir=directory)
                callback.on_before_fit(trainer, 1)
                output = model(torch.tensor([[2.], [100.]]))
                if mode == 'backward':
                    (output * torch.tensor([[2.], [100.]])).sum().backward()
                self.assertEqual(float(callback.hooks[0].mean), 51.)
                self.assertEqual(callback.hooks[0].data.shape, (2, 1))
                callback.on_after_train_batch(trainer, {}, 0)
                callback.on_after_fit(trainer)
                self.assertEqual(len(logger.histograms), 1)
                self.assertFalse(logger.closed)

    def test_failed_fit_removes_analysis_hooks_and_can_be_retried(self):
        class FailAfterBatch(Callback):
            enabled = True

            def on_after_train_batch(self, trainer, metrics, batch_idx):
                if self.enabled:
                    raise RuntimeError('batch callback failed')

        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                callback = AnalyzeCallback(mode=mode, log_dir=directory)
                stopper = FailAfterBatch()
                trainer, loader = self.logging_trainer([callback, stopper])
                trainer.logger = RecordingLogger()
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'batch callback failed'):
                        trainer.fit(loader)
                    hooks = list(callback.hooks)
                    captured = [hook.data for hook in hooks]
                    inputs, _ = next(iter(loader))
                    trainer.model.model(inputs).sum().backward()
                    self.assertTrue(all(hook.data is data for hook, data in zip(hooks, captured)))
                    self.assertFalse(any(module._forward_hooks or module._backward_hooks
                                         for module in trainer.model.modules()))
                    stopper.enabled = False
                    trainer.fit(loader)
                self.assertFalse(any(module._forward_hooks or module._backward_hooks
                                     for module in trainer.model.modules()))

    def test_before_fit_failure_removes_already_registered_analysis_hooks(self):
        class FailBeforeFit(Callback):
            def on_before_fit(self, trainer, epochs):
                raise RuntimeError('setup callback failed')

        with tempfile.TemporaryDirectory() as directory:
            callback = AnalyzeCallback(mode='forward', log_dir=directory)
            callback.priority = 1
            stopper = FailBeforeFit()
            stopper.priority = 2
            trainer, loader = self.logging_trainer([callback, stopper])
            trainer.logger = RecordingLogger()
            with self.assertRaisesRegex(RuntimeError, 'setup callback failed'):
                trainer.fit(loader)
            self.assertFalse(any(module._forward_hooks or module._backward_hooks
                                 for module in trainer.model.modules()))

    def test_after_fit_failure_removes_analysis_hooks(self):
        class FailAfterFit(Callback):
            def on_after_fit(self, trainer):
                raise RuntimeError('finish callback failed')

        with tempfile.TemporaryDirectory() as directory:
            callback = AnalyzeCallback(mode='forward', log_dir=directory)
            callback.priority = 1
            stopper = FailAfterFit()
            stopper.priority = 2
            trainer, loader = self.logging_trainer([callback, stopper])
            trainer.logger = RecordingLogger()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'finish callback failed'):
                    trainer.fit(loader)
            self.assertFalse(any(module._forward_hooks or module._backward_hooks
                                 for module in trainer.model.modules()))

    def test_close_removes_analysis_hooks_when_initialized_without_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = AnalyzeCallback(mode='forward', log_dir=directory)
            trainer, loader = self.logging_trainer([callback])
            trainer.logger = RecordingLogger()
            callback.on_before_fit(trainer, 1)
            self.assertTrue(any(module._forward_hooks for module in trainer.model.modules()))
            trainer.close()
            trainer.close()
            self.assertFalse(any(module._forward_hooks or module._backward_hooks
                                 for module in trainer.model.modules()))

    def test_non_main_analyze_creates_no_hooks_or_logger(self):
        trainer = SimpleNamespace(main_process=False)
        callback = AnalyzeCallback()
        with patch('deepepochs.callbacks.analyze.SummaryWriter') as writer:
            callback.on_before_fit(trainer, 1)
            callback.on_after_train_batch(trainer, {}, 0)
            callback.on_after_fit(trainer)
            writer.assert_not_called()
        self.assertFalse(hasattr(callback, 'hooks'))

    def test_lr_finder_counts_batches_across_epoch_boundaries(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = SimpleNamespace(main_process=True, opt=[optimizer])
        callback = LRFindCallback(max_batch=4, min_lr=0.01, max_lr=1.)
        callback.on_before_fit(trainer, 2)
        for index in (0, 1, 0):
            callback.on_before_train_batch(trainer, None, None, index)
            callback.on_after_train_batch(trainer, {'loss': torch.tensor(1.)}, index)
        callback.on_before_train_batch(trainer, None, None, 1)
        with self.assertRaises(StopLoopException):
            callback.on_after_train_batch(trainer, {'loss': torch.tensor(1.)}, 1)
        self.assertEqual(callback.batches_seen, 4)
        self.assertEqual(len(callback.losses), 4)
        self.assertTrue(all(a < b for a, b in zip(callback.lrs, callback.lrs[1:])))
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.)

    def test_lr_finder_reads_each_current_task_loss(self):
        from matplotlib import pyplot as plt
        self.addCleanup(plt.close, 'all')

        for names in (('main',), ('left', 'right', None)):
            with self.subTest(names=names):
                model = torch.nn.Linear(1, 1, bias=False)
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                finder = LRFindCallback(max_batch=3, min_lr=.01, max_lr=.1)
                # 零输入使参数梯度为零，各批损失可直接与标签平方对照。
                targets = torch.tensor([[1.], [1.], [2.], [2.], [3.], [3.]])
                if len(names) == 1:
                    loader = DataLoader(TensorDataset(torch.zeros_like(targets), targets), batch_size=2)
                    tasks = [EpochTask(loader, name=names[0])]
                else:
                    tasks = [EpochTask(DataLoader(TensorDataset(torch.zeros(2, 1), targets[2*i:2*i+2]),
                                                  batch_size=2), name=name)
                             for i, name in enumerate(names)]
                with Trainer(model, F.mse_loss, optimizer, epochs=2, device='cpu',
                             callbacks=[finder], show_info=False, log_batch=False) as trainer:
                    with contextlib.redirect_stdout(io.StringIO()), patch('deepepochs.callbacks.lr_find.plt.show'):
                        trainer.fit(train_tasks=tasks)
                self.assertEqual(finder.losses, [1., 4., 9.])
                self.assertEqual(finder.batches_seen, 3)
                self.assertEqual(len(finder.lrs), 3)
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .1)

    def test_lr_finder_non_main_obeys_main_stop_decision(self):
        class Context:
            is_distributed = True

            def broadcast_object(self, data, src):
                self_test.assertFalse(data)
                self_test.assertEqual(src, 0)
                return True

        self_test = self
        trainer = SimpleNamespace(main_process=False, distributed=Context())
        callback = LRFindCallback(max_batch=100)
        with self.assertRaises(StopLoopException):
            callback.on_after_train_batch(trainer, {'loss': torch.tensor(0.1)}, 0)
        with patch('deepepochs.callbacks.lr_find.plt.show') as show:
            callback.on_after_fit(trainer)
            show.assert_not_called()
        self.assertEqual(callback.losses, [])


if __name__ == '__main__':
    unittest.main()
