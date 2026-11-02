import contextlib
import copy
import io
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from deepepochs import Trainer
from deepepochs.callbacks import AnalyzeCallback, LogCallback


class RecordingLogger:
    def __init__(self):
        self.scalars = []
        self.histograms = []
        self.closed = False

    def add_scalar(self, tag, value, global_step):
        self.scalars.append((tag, float(value), global_step))

    def add_histogram(self, tag, value, global_step):
        self.histograms.append((tag, value, global_step))

    def close(self):
        self.closed = True


class BranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(1, 1, bias=False)
        self.right = torch.nn.Linear(1, 1, bias=False)

    def forward(self, inputs):
        layer = self.left if inputs[0, 0] < 0 else self.right
        return layer(inputs)


class SavingLinear(torch.nn.Linear):
    def forward(self, inputs):
        self.output = super().forward(inputs)
        return self.output


class AnalyzeModeTests(unittest.TestCase):
    def initialize(self, model, mode=None, **options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        if mode is not None:
            options['mode'] = mode
        callback = AnalyzeCallback(log_dir=directory.name, **options)
        logger = RecordingLogger()
        trainer = SimpleNamespace(main_process=True, running_id='test', logger=logger, model=model)
        callback.on_before_fit(trainer, 1)
        self.addCleanup(callback.remove_hooks)
        return callback, trainer, logger

    def training_setup(self, loss):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        callback = AnalyzeCallback(log_dir=directory.name)
        model = SavingLinear(1, 1)
        trainer = Trainer(model, loss, torch.optim.SGD(model.parameters(), lr=.1),
                          epochs=1, callbacks=[callback], device='cpu',
                          show_info=False, log_batch=False, running_id='test')
        trainer.logger = RecordingLogger()
        self.addCleanup(trainer.close)
        inputs = torch.tensor([[1.], [2.]])
        loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
        return callback, trainer, model, loader

    def assert_removed(self, callback, model, output):
        self.assertFalse(any(module._forward_hooks or module._backward_hooks
                             for module in model.modules()))
        self.assertFalse(output._backward_hooks)
        self.assertTrue(all(not hook.tensor_hooks for hook in callback.hooks))
        output.sum().backward()
        self.assertTrue(all(not hasattr(hook, 'data') for hook in callback.hooks))

    def test_sampling_interval_must_be_a_positive_integer(self):
        for interval in (0, -1, 1.5, True, '2'):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, 'every_n_batches'):
                AnalyzeCallback(every_n_batches=interval)

    def test_unknown_layer_names_are_rejected_without_installing_hooks(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
        for name in ('missing', '0.*', '0.weight'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '未知层名'):
                self.initialize(model, layers=['0', name])
            self.assertFalse(any(module._forward_hooks for module in model.modules()))
        with self.assertRaisesRegex(TypeError, 'layers'):
            AnalyzeCallback(layers='0')

    def test_unsampled_batches_skip_statistics_and_tensor_hook_registration(self):
        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode):
                model = SavingLinear(1, 1)
                callback, trainer, logger = self.initialize(model, mode, every_n_batches=2)
                inputs = torch.tensor([[1.], [2.]])
                callback.on_before_train_batch(trainer, inputs, None, 0)
                model(inputs).sum().backward()
                callback.on_after_train_batch(trainer, {}, 0)
                self.assertEqual(len(logger.histograms), 1)
                callback.on_before_train_batch(trainer, inputs, None, 1)
                with contextlib.ExitStack() as stack:
                    for operation in ('detach', 'clone', 'mean', 'std', 'register_hook'):
                        stack.enter_context(patch.object(torch.Tensor, operation,
                            side_effect=AssertionError(f'unsampled {operation}')))
                    output = model(inputs)
                    self.assertFalse(output._backward_hooks)
                    self.assertTrue(all(not hook.tensor_hooks for hook in callback.hooks))
                    output.sum().backward()
                    callback.on_after_train_batch(trainer, {}, 1)
                self.assertEqual(callback.global_step, 2)
                self.assertEqual(len(logger.histograms), 1)
                callback.on_before_train_batch(trainer, inputs, None, 2)
                model(inputs).sum().backward()
                callback.on_after_train_batch(trainer, {}, 2)
                self.assertEqual([step for _, _, step in logger.histograms], [1, 3])

    def test_exact_parent_and_leaf_selection_preserves_gradients_and_updates(self):
        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode):
                model = torch.nn.Sequential(
                    torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(inplace=True)),
                    torch.nn.Linear(3, 1))
                reference = copy.deepcopy(model)
                callback, trainer, logger = self.initialize(model, mode, layers=['0', '0.1'])
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
                inputs = torch.tensor([[1., 2.], [-2., .5]])
                target = torch.tensor([[1.], [0.]])
                callback.on_before_train_batch(trainer, inputs, target, 0)
                output, expected = model(inputs), reference(inputs)
                torch.testing.assert_close(output, expected)
                F.mse_loss(output, target).backward()
                F.mse_loss(expected, target).backward()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad)
                optimizer.step()
                reference_optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected)
                callback.on_after_train_batch(trainer, {}, 0)
                self.assertEqual(len(callback.hooks), 2)
                self.assertTrue(callback.hooks[0].name.startswith('0-Sequential('))
                self.assertTrue(callback.hooks[1].name.startswith('0.1-ReLU('))
                self.assertEqual([tag for tag, _, _ in logger.histograms],
                                 [f'{mode.upper()}-hist-{hook.name}' for hook in callback.hooks])

    def test_explicit_root_selection_keeps_first_tensor_from_parent_tuple(self):
        class TupleParent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.layer.weight.fill_(1.)

            def forward(self, inputs):
                output = self.layer(inputs)
                return output, output * 2.

        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode):
                callback, trainer, logger = self.initialize(TupleParent(), mode, layers=[''])
                inputs = torch.tensor([[2.], [100.]])
                output, other = trainer.model(inputs)
                (output * inputs).sum().backward()
                callback.on_after_train_batch(trainer, {}, 0)
                self.assertEqual(len(callback.hooks), 1)
                self.assertTrue(callback.hooks[0].name.startswith('<root>-TupleParent('))
                torch.testing.assert_close(logger.histograms[0][1], inputs)

    def test_explicit_names_use_original_model_under_compile(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
        compiled = torch.compile(model, backend='eager')
        with tempfile.TemporaryDirectory() as directory:
            callback = AnalyzeCallback(mode='forward', layers=['0'], log_dir=directory)
            trainer = Trainer(compiled, F.mse_loss, torch.optim.SGD(model.parameters(), lr=.1),
                              epochs=1, callbacks=[callback], device='cpu',
                              show_info=False, log_batch=False, running_id='compiled')
            trainer.logger = RecordingLogger()
            inputs = torch.tensor([[1.], [2.]])
            loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.fit(loader)
                self.assertTrue(callback.hooks[0].name.startswith('0-Linear('))
                self.assertEqual(len(trainer.logger.histograms), 1)
            finally:
                trainer.close()

    def test_sampling_phase_continues_across_fits_and_state_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = AnalyzeCallback(mode='forward', every_n_batches=2, log_dir=directory)
            model = torch.nn.Linear(1, 1)
            trainer = Trainer(model, F.mse_loss, torch.optim.SGD(model.parameters(), lr=.1),
                              epochs=1, callbacks=[callback], device='cpu',
                              show_info=False, log_batch=False, running_id='sampling')
            trainer.logger = RecordingLogger()
            inputs = torch.arange(1., 7.).view(-1, 1)
            loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.fit(loader)
                    trainer.fit(loader, epochs=1)
                self.assertEqual(callback.global_step, 6)
                self.assertEqual([step for _, _, step in trainer.logger.histograms], [1, 3, 5])
                state = callback.state_dict()
                self.assertEqual(state, {'global_step': 6})
                restored = AnalyzeCallback(mode='forward', every_n_batches=2, log_dir=directory)
                restored.load_state_dict(state)
                self.assertFalse(hasattr(restored, 'hooks'))
                self.assertFalse(hasattr(restored, 'logger'))
                restored.on_before_fit(trainer, 1)
                for index in range(3):
                    restored.on_before_train_batch(trainer, inputs, None, index)
                    model(inputs).sum().backward()
                    restored.on_after_train_batch(trainer, {}, index)
                restored.on_after_fit(trainer)
                self.assertEqual(restored.global_step, 9)
                self.assertEqual([step for _, _, step in trainer.logger.histograms], [1, 3, 5, 7, 9])
            finally:
                trainer.close()

    def test_default_backward_preserves_inplace_training_gradients_and_updates(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(3, 1))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1., -1.], [.5, .25], [-1., -.5]]))
            model[0].bias.copy_(torch.tensor([.2, .3, -.4]))
            model[2].weight.copy_(torch.tensor([[.5, -.3, .2]]))
            model[2].bias.fill_(.1)
        reference = copy.deepcopy(model)
        callback, trainer, logger = self.initialize(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
        batches = [(torch.tensor([[-2., 1.], [1., 2.]]), torch.tensor([[0.], [1.]])),
                   (torch.tensor([[2., -1.], [-1., -2.]]), torch.tensor([[1.], [0.]]))]
        for index, (inputs, target) in enumerate(batches):
            callback.on_before_train_batch(trainer, inputs, target, index)
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            output = model(inputs)
            reference_output = reference(inputs)
            torch.testing.assert_close(output, reference_output)
            F.mse_loss(output, target).backward()
            F.mse_loss(reference_output, target).backward()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected)
            callback.on_after_train_batch(trainer, {}, index)
        self.assertEqual(len(logger.histograms), 4)
        self.assertTrue(all(not hook.tensor_hooks for hook in callback.hooks))

    def test_forward_histogram_preserves_output_before_inplace_relu(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False),
                                    torch.nn.ReLU(inplace=True))
        with torch.no_grad():
            model[0].weight.fill_(-1.)
        callback, trainer, logger = self.initialize(model, 'forward')
        output = model(torch.tensor([[1.], [3.]]))
        self.assertEqual(float(output.detach().sum()), 0.)
        callback.on_after_train_batch(trainer, {}, 0)
        hook = callback.hooks[0]
        self.assertEqual(float(hook.mean), -2.)
        torch.testing.assert_close(hook.data, torch.tensor([[-1.], [-3.]]))
        histogram = next(data for tag, data, step in logger.histograms
                         if tag == f'FORWARD-hist-{hook.name}')
        torch.testing.assert_close(histogram, hook.data)
        self.assertEqual(float(histogram.mean()), -2.)

    def test_backward_histogram_preserves_gradient_before_later_hook_mutation(self):
        model = torch.nn.Linear(1, 1, bias=False)
        callback, trainer, logger = self.initialize(model)
        output = model(torch.tensor([[1.], [2.]]))
        handle = output.register_hook(lambda gradient: gradient.mul_(2.))
        self.addCleanup(handle.remove)
        weights = torch.tensor([[2.], [100.]])
        (output * weights).sum().backward()
        callback.on_after_train_batch(trainer, {}, 0)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[404.]]))
        torch.testing.assert_close(callback.hooks[0].data, weights)
        torch.testing.assert_close(logger.histograms[0][1], weights)
        self.assertEqual(float(callback.hooks[0].mean), 51.)

    def test_alternating_branches_record_only_layers_used_in_the_current_batch(self):
        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode):
                model = BranchModel()
                callback, trainer, logger = self.initialize(model, mode)
                left, right = callback.hooks
                for index, value in enumerate((-1., 1., -2.)):
                    if index:
                        # 验证阶段执行另一条分支，不能混入下一训练 batch。
                        with torch.no_grad():
                            model(torch.tensor([[-value], [-2. * value]]))
                    inputs = torch.tensor([[value], [2. * value]])
                    callback.on_before_train_batch(trainer, inputs, None, index)
                    model.zero_grad()
                    model(inputs).sum().backward()
                    callback.on_after_train_batch(trainer, {}, index)
                    expected = left if value < 0 else right
                    prefix = mode.upper()
                    step = index + 1
                    self.assertEqual([tag for tag, data, number in logger.histograms if number == step],
                                     [f'{prefix}-hist-{expected.name}'])
                    self.assertEqual([tag for tag, value, number in logger.scalars if number == step],
                                     [f'{prefix}-mean-{expected.name}', f'{prefix}-std-{expected.name}'])
                self.assertEqual(len(logger.histograms), 3)

    def test_graph_tracing_does_not_record_evaluation_branch_as_training_output(self):
        class ModeBranchModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.training_branch = torch.nn.Linear(1, 1, bias=False)
                self.evaluation_branch = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.training_branch.weight.fill_(1.)
                    self.evaluation_branch.weight.fill_(10.)

            def forward(self, inputs):
                branch = self.training_branch if self.training else self.evaluation_branch
                return branch(inputs)

        with tempfile.TemporaryDirectory() as directory:
            analysis = AnalyzeCallback(mode='forward', log_dir=directory)
            logging = LogCallback(log_graph=True, log_dir=directory)
            model = ModeBranchModel()
            trainer = Trainer(model, F.mse_loss, torch.optim.SGD(model.parameters(), lr=.1),
                              epochs=1, callbacks=[analysis, logging], device='cpu',
                              show_info=False, log_batch=False, running_id='graph')
            writer = SummaryWriter(log_dir=directory)
            trainer.logger = writer
            inputs = torch.tensor([[1.], [2.]])
            loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
            try:
                with patch.object(writer, 'add_histogram', wraps=writer.add_histogram) as histogram:
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit(loader)
                    self.assertTrue(logging.graph_saved)
                    self.assertTrue(model.training)
                    training_hook = analysis.hooks[0]
                    self.assertEqual([call.args[0] for call in histogram.call_args_list],
                                     [f'FORWARD-hist-{training_hook.name}'])
                    torch.testing.assert_close(histogram.call_args.args[1], inputs)
            finally:
                trainer.close()

    def test_bfloat16_autocast_writes_finite_events_without_changing_training(self):
        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(inplace=True),
                                            torch.nn.Linear(3, 1))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(.25)
                reference = copy.deepcopy(model)
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
                writer = SummaryWriter(log_dir=directory)
                callback = AnalyzeCallback(mode=mode, log_dir=directory)
                trainer = SimpleNamespace(main_process=True, running_id='bf16',
                                          logger=writer, model=model)
                callback.on_before_fit(trainer, 1)
                batches = [(torch.tensor([[1., 2.], [-1., .5]]), torch.tensor([[0.], [1.]])),
                           (torch.tensor([[-2., -1.], [.5, 2.]]), torch.tensor([[1.], [0.]]))]
                try:
                    for index, (inputs, target) in enumerate(batches):
                        callback.on_before_train_batch(trainer, inputs, target, index)
                        optimizer.zero_grad()
                        reference_optimizer.zero_grad()
                        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                            output = model(inputs)
                            reference_output = reference(inputs)
                            loss = F.mse_loss(output, target)
                            reference_loss = F.mse_loss(reference_output, target)
                        self.assertEqual(output.dtype, torch.bfloat16)
                        torch.testing.assert_close(output, reference_output)
                        loss.backward()
                        reference_loss.backward()
                        for actual, expected in zip(model.parameters(), reference.parameters()):
                            torch.testing.assert_close(actual.grad, expected.grad)
                        optimizer.step()
                        reference_optimizer.step()
                        for actual, expected in zip(model.parameters(), reference.parameters()):
                            torch.testing.assert_close(actual, expected)
                        callback.on_after_train_batch(trainer, {}, index)
                        self.assertTrue(all(hook.data.dtype == torch.float32 for hook in callback.hooks))
                finally:
                    callback.on_after_fit(trainer)
                    writer.close()

                events = EventAccumulator(directory, size_guidance={'histograms': 0}).Reload()
                prefix = mode.upper()
                self.assertEqual(len(events.Tags()['scalars']), 2 * len(callback.hooks))
                self.assertEqual(len(events.Tags()['histograms']), len(callback.hooks))
                for hook in callback.hooks:
                    for statistic in ('mean', 'std'):
                        scalars = events.Scalars(f'{prefix}-{statistic}-{hook.name}')
                        self.assertEqual([event.step for event in scalars], [1, 2])
                        self.assertTrue(all(math.isfinite(event.value) for event in scalars))
                    histograms = events.Histograms(f'{prefix}-hist-{hook.name}')
                    self.assertEqual([event.step for event in histograms], [1, 2])
                    for event in histograms:
                        histogram = event.histogram_value
                        values = [histogram.min, histogram.max, histogram.num,
                                  histogram.sum, histogram.sum_squares,
                                  *histogram.bucket, *histogram.bucket_limit]
                        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_tuple_outputs_keep_first_tensor_statistics(self):
        class TupleLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, inputs):
                return inputs * self.weight, inputs * self.weight * 2.

        for mode in ('forward', 'backward'):
            with self.subTest(mode=mode):
                callback, trainer, logger = self.initialize(TupleLayer(), mode)
                inputs = torch.tensor([[2.], [100.]])
                output, other = trainer.model(inputs)
                (output * inputs).sum().backward()
                callback.on_after_train_batch(trainer, {}, 0)
                torch.testing.assert_close(logger.histograms[0][1], inputs)
                self.assertEqual(float(callback.hooks[0].mean), 51.)

    def test_after_fit_removes_tensor_hooks_from_outputs_waiting_for_backward(self):
        model = torch.nn.Linear(1, 1)
        callback, trainer, logger = self.initialize(model)
        output = model(torch.tensor([[1.], [2.]]))
        self.assertTrue(output._backward_hooks)
        callback.on_after_fit(trainer)
        self.assert_removed(callback, model, output)
        self.assertFalse(logger.closed)

    def test_failed_fit_removes_tensor_hooks_from_outputs_waiting_for_backward(self):
        def failing_loss(output, target):
            raise RuntimeError('loss failed')

        callback, trainer, model, loader = self.training_setup(failing_loss)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'loss failed'):
                trainer.fit(loader)
        self.assert_removed(callback, model, model.output)

    def test_close_removes_tensor_hooks_from_outputs_waiting_for_backward(self):
        callback, trainer, model, loader = self.training_setup(F.mse_loss)
        callback.on_before_fit(trainer, 1)
        output = model(torch.tensor([[1.], [2.]]))
        self.assertTrue(output._backward_hooks)
        logger = trainer.logger
        trainer.close()
        trainer.close()
        self.assert_removed(callback, model, output)
        self.assertTrue(logger.closed)


if __name__ == '__main__':
    unittest.main()
