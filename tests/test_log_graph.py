import contextlib
from datetime import timedelta
import io
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData

from deepepochs import CheckCallback, LogCallback, Trainer


class DictLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, *, x, y):
        return {'loss': F.mse_loss(self.linear(x), y)}


class PairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, offset):
        return self.linear(x) + offset


class BufferedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(0.25)
        self.register_buffer('scale', torch.ones(1))

    def forward(self, x):
        return self.linear(x) * self.scale


class TraceControlModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.5)
        self.dropout.eval()
        self.reject_trace = False

    def forward(self, x):
        if self.reject_trace and torch.jit.is_tracing():
            raise RuntimeError('trace unavailable')
        return self.dropout(self.linear(x))


class RandomForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.reject_trace = False

    def forward(self, x):
        random.random()
        np.random.rand()
        torch.rand(())
        noise = torch.rand_like(x)
        if self.reject_trace and torch.jit.is_tracing():
            raise RuntimeError('random trace unavailable')
        return self.linear(x) + self.dropout(noise) * 0.1


def _seed_random(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _random_draws(device):
    values = {'python': random.random(), 'numpy': float(np.random.rand()),
              'torch_cpu': torch.rand(3)}
    if torch.device(device).type == 'cuda':
        values['torch_cuda'] = torch.rand(3, device=device).cpu()
    return values


def _trainer(model, callback, loss=F.mse_loss, **kwargs):
    trainer = Trainer(model, loss, torch.optim.SGD(model.parameters(), lr=0.1),
                      device='cpu', epochs=1, callbacks=[callback],
                      show_info=False, log_batch=False, running_id='graph', **kwargs)
    return trainer


def _graph(log_directory):
    return EventAccumulator(str(log_directory)).Reload().Graph()


def _ddp_graph_worker(rank, init_path, log_directory):
    torch.set_num_threads(1)
    os.environ['LOCAL_RANK'] = str(rank)
    dist.init_process_group('gloo', init_method='file://' + init_path,
                            rank=rank, world_size=2, timeout=timedelta(seconds=15))
    output = io.StringIO()
    try:
        weights = []
        for save_graph in (False, True):
            directory = str(Path(log_directory) / ('enabled' if save_graph else 'baseline'))
            callback = LogCallback(log_dir=directory, log_graph=save_graph)
            model = BufferedModel()
            trainer = _trainer(model, callback)
            writer = None
            try:
                inputs = torch.arange(1., 9.).reshape(-1, 1) / 8
                loader = DataLoader(TensorDataset(inputs, 2 * inputs), batch_size=2)
                with contextlib.redirect_stdout(output):
                    trainer.fit(loader)
                weights.append(model.linear.weight.detach().clone())
                assert callback.graph_saved == (save_graph and rank == 0)
                assert not torch.equal(weights[-1], torch.tensor([[0.25]]))
                if rank == 0:
                    writer = trainer.logger
                    worker = writer.file_writer.event_writer._async_writer._worker
                    assert worker.is_alive()
                else:
                    assert getattr(callback, 'logger', None) is None
            finally:
                trainer.close()
                if writer is not None:
                    assert not worker.is_alive()
                    assert trainer.logger is None
                    assert callback.logger is None
        torch.testing.assert_close(weights[0], weights[1])
        assert '模型结构图保存失败' not in output.getvalue(), output.getvalue()
        assert 'No graph saved' not in output.getvalue(), output.getvalue()
    finally:
        dist.destroy_process_group()


class LogGraphTests(unittest.TestCase):
    def assert_random_draws_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
            else:
                self.assertEqual(actual[key], value)

    def assert_random_training_resume(self, device):
        with tempfile.TemporaryDirectory() as directory:
            def train(root, epochs, log_graph, running_id='full', resume=False):
                logger = LogCallback(log_dir=str(root / 'logs'), log_graph=log_graph)
                checker = CheckCallback('loss', on_stage='train', save_best=False,
                                        ckpt_dir=str(root / 'checkpoints'))
                model = RandomForwardModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.7)
                inputs = torch.arange(1., 9.).reshape(-1, 1) / 8
                loader = DataLoader(TensorDataset(inputs, 2 * inputs), batch_size=2,
                                    shuffle=True, generator=torch.Generator().manual_seed(31))
                with Trainer(model, F.mse_loss, optimizer, device=device, epochs=epochs,
                             callbacks=[logger, checker], resume=resume, running_id=running_id,
                             show_info=False, log_batch=False) as trainer:
                    with contextlib.redirect_stdout(io.StringIO()):
                        history = trainer.fit(loader)
                    self.assertEqual(logger.graph_saved, log_graph)
                    weights = {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()}
                    momentum = [state['momentum_buffer'].detach().cpu().clone()
                                for state in optimizer.state.values()]
                    return weights, momentum, _random_draws(device), history

            root = Path(directory)
            _seed_random(79)
            baseline = train(root / 'baseline', 3, False)
            _seed_random(79)
            continuous = train(root / 'continuous', 3, True)
            _seed_random(79)
            train(root / 'split', 1, True, running_id='saved')
            _seed_random(917)
            resumed = train(root / 'split', 3, True, running_id='continued', resume='saved')
            for actual in (continuous, resumed):
                for name, value in baseline[0].items():
                    torch.testing.assert_close(actual[0][name], value, atol=0, rtol=0)
                for actual_buffer, expected_buffer in zip(actual[1], baseline[1]):
                    torch.testing.assert_close(actual_buffer, expected_buffer, atol=0, rtol=0)
                self.assert_random_draws_equal(actual[2], baseline[2])
            self.assertEqual(resumed[3]['train']['loss'], continuous[3]['train']['loss'][1:])

    def assert_failed_random_graph(self, device):
        with tempfile.TemporaryDirectory() as directory:
            callback = LogCallback(log_dir=directory, log_graph=True)
            model = RandomForwardModel()
            model.reject_trace = True
            model.dropout.eval()
            with Trainer(model, F.mse_loss, torch.optim.SGD(model.parameters(), lr=0.1),
                         device=device, callbacks=[callback], show_info=False, log_batch=False) as trainer:
                callback.on_before_fit(trainer, 1)
                modes = [module.training for module in model.modules()]
                forward = model.forward
                _seed_random(63)
                expected = _random_draws(device)
                _seed_random(63)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    callback.on_before_train_batch(trainer, torch.ones(2, 1, device=device), None, 0)
                self.assertFalse(callback.graph_saved)
                self.assertIn('random trace unavailable', output.getvalue())
                self.assert_random_draws_equal(_random_draws(device), expected)
                self.assertEqual([module.training for module in model.modules()], modes)
                self.assertEqual(model.forward, forward)

    def test_random_graph_preserves_continuous_and_resumed_training(self):
        self.assert_random_training_resume('cpu')

    def test_failed_random_graph_restores_streams_and_modes(self):
        self.assert_failed_random_graph('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_random_graph_preserves_continuous_and_resumed_cuda_training(self):
        self.assert_random_training_resume('cuda:0')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_failed_random_graph_restores_cuda_streams_and_modes(self):
        self.assert_failed_random_graph('cuda:0')

    def test_bf16_metrics_write_train_val_and_test_hparams_scalars(self):
        observed = []

        def bf16_mean(preds, targets):
            value = preds.max(-1).values.mean()
            self.assertEqual(value.dtype, torch.bfloat16)
            observed.append(float(value))
            return value

        with tempfile.TemporaryDirectory() as directory:
            callback = LogCallback(log_dir=directory)
            model = nn.Linear(1, 2, bias=False)
            model.weight.data.copy_(torch.tensor([[0.25], [-0.5]]))
            before = model.weight.detach().clone()
            trainer = _trainer(model, callback, loss=F.cross_entropy, precision='bf16',
                               metrics=[bf16_mean], hyper_params={'precision': 'bf16', 'batch_size': 2})
            inputs = torch.arange(1., 5.).reshape(-1, 1) / 4
            loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 0, 1])), batch_size=2)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    history = trainer.fit(loader, loader)
                    result = trainer.test(loader)
                self.assertFalse(torch.equal(before, model.weight))
            finally:
                trainer.close()
            self.assertEqual(len(observed), 9)
            events = EventAccumulator(str(Path(directory) / 'graph')).Reload()
            for stage, offset in (('train', 0), ('val', 3)):
                with self.subTest(stage=stage):
                    self.assertEqual([event.value for event in events.Scalars(f'bf16_mean/batch/{stage}')],
                                     observed[offset:offset + 2])
                    self.assertEqual([event.value for event in events.Scalars(f'bf16_mean/epoch/{stage}')],
                                     [observed[offset + 2]])
                    self.assertEqual(float(history[stage]['bf16_mean'][0]), observed[offset + 2])
            self.assertEqual([event.value for event in events.Scalars('bf16_mean')], [observed[8]])
            self.assertEqual(float(result['bf16_mean']), observed[8])
            metadata = events.PluginTagToContent('hparams')
            session = HParamsPluginData.FromString(metadata['_hparams_/session_start_info'])
            self.assertEqual(session.session_start_info.hparams['precision'].string_value, 'bf16')
            self.assertEqual(session.session_start_info.hparams['batch_size'].number_value, 2)

    def test_dict_inputs_and_outputs_train_and_write_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = LogCallback(log_dir=directory, log_graph=True)
            model = DictLossModel()
            trainer = _trainer(model, callback, loss=None)
            before = model.linear.weight.detach().clone()
            inputs = torch.arange(1., 5.).reshape(-1, 1)
            loader = DataLoader([{'x': x, 'y': 2 * x} for x in inputs], batch_size=2)
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    trainer.fit(loader)
                self.assertTrue(callback.graph_saved)
                self.assertNotIn('模型结构图保存失败', output.getvalue())
                self.assertFalse(torch.equal(before, model.linear.weight))
                self.assertTrue(model.training)
            finally:
                trainer.close()
            ops = {node.op for node in _graph(Path(directory) / 'graph').node}
            self.assertIn('aten::linear', ops)
            self.assertIn('aten::mse_loss', ops)
            self.assertIn('prim::DictConstruct', ops)

    def test_tuple_example_input_keeps_positional_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            example = (torch.ones(3, 1), torch.full((3, 1), 2.))
            callback = LogCallback(log_dir=directory, log_graph=True, example_input=example)
            trainer = _trainer(PairModel(), callback)
            inputs = torch.ones(4, 1)
            loader = DataLoader(TensorDataset(inputs, 2 * inputs, 3 * inputs), batch_size=2)
            try:
                callback.on_before_fit(trainer, 1)
                with patch.object(trainer.logger, 'add_graph', wraps=trainer.logger.add_graph) as add_graph:
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit(loader)
                    add_graph.assert_called_once()
                    self.assertIs(add_graph.call_args.args[1], example)
                self.assertTrue(callback.graph_saved)
            finally:
                trainer.close()
            graph = _graph(Path(directory) / 'graph')
            self.assertIn('aten::add', {node.op for node in graph.node})
            self.assertEqual(sum(node.op == 'IO Node' and node.name.startswith('input/')
                                 for node in graph.node), 2)

    def test_compiled_model_is_unwrapped_for_graph_only(self):
        with tempfile.TemporaryDirectory() as directory:
            model = BufferedModel()
            compiled = torch.compile(model, backend='eager')
            callback = LogCallback(log_dir=directory, log_graph=True)
            trainer = _trainer(compiled, callback)
            try:
                callback.on_before_fit(trainer, 1)
                with patch.object(trainer.logger, 'add_graph', wraps=trainer.logger.add_graph) as add_graph:
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit(DataLoader(TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)),
                                               batch_size=2))
                    self.assertIs(add_graph.call_args.args[0], model)
                self.assertIs(trainer.model.model, compiled)
                self.assertTrue(callback.graph_saved)
                self.assertTrue(model.training)
            finally:
                trainer.close()
            self.assertIn('aten::linear', {node.op for node in _graph(Path(directory) / 'graph').node})

    def test_graph_preserves_forward_and_mixed_training_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = LogCallback(log_dir=directory, log_graph=True)
            model = TraceControlModel()
            trainer = _trainer(model, callback)
            inputs = torch.ones(2, 1)
            forward = model.forward
            modes = [module.training for module in model.modules()]
            expected = model(inputs).detach()
            try:
                callback.on_before_fit(trainer, 1)
                callback.on_before_train_batch(trainer, inputs, None, 0)
                self.assertTrue(callback.graph_saved)
                self.assertEqual(model.forward, forward)
                self.assertEqual([module.training for module in model.modules()], modes)
                torch.testing.assert_close(model(inputs), expected)
            finally:
                trainer.close()

    def test_failed_trace_retries_once_on_next_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = LogCallback(log_dir=directory, log_graph=True)
            model = TraceControlModel()
            model.reject_trace = True
            trainer = _trainer(model, callback)
            inputs = torch.ones(6, 1)
            loader = DataLoader(TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=2)
            try:
                callback.on_before_fit(trainer, 1)
                output = io.StringIO()
                with patch.object(trainer.logger, 'add_graph', wraps=trainer.logger.add_graph) as add_graph:
                    with contextlib.redirect_stdout(output):
                        trainer.fit(loader, epochs=1)
                    add_graph.assert_called_once()
                    self.assertFalse(callback.graph_saved)
                    self.assertEqual(output.getvalue().count('模型结构图保存失败'), 1)
                    self.assertIn('trace unavailable', output.getvalue())
                    self.assertTrue(all(module.training for module in model.modules()))
                    trainer.logger.flush()
                    with self.assertRaises(ValueError):
                        _graph(Path(directory) / 'graph')
                    model.reject_trace = False
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit(loader, epochs=1)
                    self.assertEqual(add_graph.call_count, 2)
                    self.assertTrue(callback.graph_saved)
                    self.assertTrue(all(module.training for module in model.modules()))
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit(loader, epochs=1)
                    self.assertEqual(add_graph.call_count, 2)
            finally:
                trainer.close()
            self.assertIn('aten::linear', {node.op for node in _graph(Path(directory) / 'graph').node})

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'requires CPU/Gloo')
    def test_two_rank_ddp_trains_and_writes_graph_without_trace_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_ddp_graph_worker, args=(str(Path(directory) / 'init'), directory),
                     nprocs=2, join=True)
            graph = _graph(Path(directory) / 'enabled' / 'graph')
            self.assertIn('aten::linear', {node.op for node in graph.node})
            with self.assertRaises(ValueError):
                _graph(Path(directory) / 'baseline' / 'graph')


if __name__ == '__main__':
    unittest.main()
