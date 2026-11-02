"""无标签流式预测、独立输出与临时评估权重。"""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data._utils.collate import default_collate

from deepepochs import CheckCallback, Trainer
from deepepochs.callbacks.check import save_state


class StructuredModel(nn.Module):
    def __init__(self, output_kind):
        super().__init__()
        self.output_kind = output_kind

    def forward(self, x, offset=None):
        value = x if offset is None else x + offset
        if self.output_kind == 'tuple':
            return value, value.to(torch.int64)
        if self.output_kind == 'list':
            return [value, value > 0]
        if self.output_kind == 'dict':
            return {'value': value, 'index': value.to(torch.int64)}
        if self.output_kind == 'nested':
            return {'parts': (value, [value.to(torch.int64), {'positive': value > 0}])}
        return value


class ReusedBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('output', torch.zeros(1, 1, dtype=torch.float64))

    def forward(self, x):
        self.output.copy_(x)
        return self.output


class PredictWithIds(Trainer):
    def predict_step(self, batch_x):
        return {'sample_id': batch_x['sample_id'], 'value': self.model(batch_x['x'])}


class FailSecondPrediction(Trainer):
    def predict_step(self, batch_x):
        self.prediction_count = getattr(self, 'prediction_count', 0) + 1
        if self.prediction_count == 2:
            raise RuntimeError('prediction failed')
        return super().predict_step(batch_x)


class ObserveContexts(nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)
        self.contexts = []

    def forward(self, x):
        self.contexts.append((torch.is_grad_enabled(), torch.is_autocast_cpu_enabled()))
        return super().forward(x)


class PredictTests(unittest.TestCase):
    def trainer(self, model, trainer_type=Trainer, **kwargs):
        trainer = trainer_type(model, device='cpu', show_info=False, log_batch=False, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_predict_accepts_unlabelled_inputs_and_preserves_output_structures(self):
        values = torch.tensor([[1.], [2.]], dtype=torch.float64)
        offset = torch.full_like(values, 3.)
        for input_kind, output_kind in (('tensor', 'tensor'), ('tuple', 'tuple'),
                                        ('list', 'list'), ('dict', 'dict'), ('dict', 'nested')):
            with self.subTest(inputs=input_kind, outputs=output_kind):
                if input_kind == 'tensor':
                    loader = DataLoader(values, batch_size=2)
                    expected = values
                elif input_kind == 'dict':
                    loader = DataLoader([{'x': x, 'offset': y} for x, y in zip(values, offset)], batch_size=2)
                    expected = values + offset
                else:
                    collate = (lambda items: tuple(default_collate(items))) if input_kind == 'tuple' else default_collate
                    loader = DataLoader(TensorDataset(values, offset), batch_size=2, collate_fn=collate)
                    expected = values + offset
                trainer = self.trainer(StructuredModel(output_kind))
                outputs = list(trainer.predict(loader))
                self.assertEqual(len(outputs), 1)
                result = outputs[0]
                if output_kind == 'tuple':
                    self.assertIsInstance(result, tuple)
                    value, integers = result
                    torch.testing.assert_close(integers, expected.to(torch.int64))
                elif output_kind == 'list':
                    self.assertIsInstance(result, list)
                    value, mask = result
                    torch.testing.assert_close(mask, expected > 0)
                elif output_kind == 'dict':
                    self.assertEqual(set(result), {'value', 'index'})
                    value = result['value']
                    torch.testing.assert_close(result['index'], expected.to(torch.int64))
                elif output_kind == 'nested':
                    self.assertIsInstance(result['parts'], tuple)
                    value, nested = result['parts']
                    self.assertIsInstance(nested, list)
                    torch.testing.assert_close(nested[0], expected.to(torch.int64))
                    torch.testing.assert_close(nested[1]['positive'], expected > 0)
                else:
                    value = result
                torch.testing.assert_close(value, expected)
                self.assertEqual(value.device.type, 'cpu')
                self.assertEqual(value.dtype, torch.float64)
                self.assertFalse(value.requires_grad)

    def test_parameterless_identity_predicts_without_loss_or_optimizer(self):
        trainer = self.trainer(nn.Identity())
        values = torch.tensor([[1., 2.], [3., 4.]])
        result = list(trainer.predict(DataLoader(values, batch_size=1)))
        torch.testing.assert_close(torch.cat(result), values)
        self.assertEqual(len(trainer.opt), 0)

    def test_cpu_predictions_do_not_alias_reused_model_buffer(self):
        model = ReusedBuffer()
        trainer = self.trainer(model)
        values = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
        outputs = list(trainer.predict(DataLoader(values, batch_size=1)))
        with torch.no_grad():
            model.output.fill_(99.)
        for output, expected in zip(outputs, values):
            torch.testing.assert_close(output, expected.reshape(1, 1))
            self.assertNotEqual(output.data_ptr(), model.output.data_ptr())
        outputs[0].zero_()
        self.assertEqual(float(outputs[1]), 2.)

    def test_custom_predict_step_keeps_sample_ids(self):
        trainer = self.trainer(nn.Identity(), trainer_type=PredictWithIds)
        loader = DataLoader([{'x': torch.tensor([value]), 'sample_id': name}
                             for value, name in ((1., 'sample-a'), (2., 'sample-b'))], batch_size=1)
        outputs = list(trainer.predict(loader))
        self.assertEqual([output['sample_id'] for output in outputs], [('sample-a',), ('sample-b',)])
        torch.testing.assert_close(torch.cat([output['value'] for output in outputs]), torch.tensor([[1.], [2.]]))

    def test_grad_and_autocast_contexts_do_not_leak_between_yields(self):
        for outer_grad in (True, False):
            with self.subTest(outer_grad=outer_grad):
                model = ObserveContexts()
                trainer = self.trainer(model, precision='bf16')
                loader = DataLoader(torch.ones(2, 1), batch_size=1)
                with torch.set_grad_enabled(outer_grad), contextlib.closing(trainer.predict(loader)) as stream:
                    for output in stream:
                        self.assertEqual(torch.is_grad_enabled(), outer_grad)
                        self.assertFalse(torch.is_autocast_cpu_enabled())
                        self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(model.contexts, [(False, True), (False, True)])
        model = ObserveContexts()
        trainer = self.trainer(model)
        with torch.autocast('cpu', dtype=torch.bfloat16), contextlib.closing(
                trainer.predict(DataLoader(torch.ones(1, 1), batch_size=1))) as stream:
            next(stream)
            self.assertTrue(torch.is_autocast_cpu_enabled())
        self.assertFalse(torch.is_autocast_cpu_enabled())

    def test_closing_and_prediction_errors_restore_weights_modes_stage_and_task(self):
        for ending in ('close', 'error'):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory:
                model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Dropout())
                trainer_type = FailSecondPrediction if ending == 'error' else Trainer
                trainer = self.trainer(model, trainer_type=trainer_type)
                with torch.no_grad():
                    model[0].weight.fill_(.5)
                path = Path(directory) / 'weights.ckpt'
                save_state(trainer, trainer.model, trainer.opt, path, epoch=0)
                with torch.no_grad():
                    model[0].weight.fill_(2.)
                model.train()
                model[1].eval()
                original_modes = [module.training for module in model.modules()]
                original_task = object()
                trainer.current_task = original_task
                trainer.model.stage = 'train'
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.closing(trainer.predict(DataLoader(torch.ones(2, 1), batch_size=1), weights=path)) as stream:
                        first = next(stream)
                        torch.testing.assert_close(first, torch.tensor([[.5]]))
                        self.assertFalse(model.training)
                        self.assertEqual(trainer.model.stage, 'predict')
                        self.assertIsNone(trainer.current_task)
                        if ending == 'error':
                            with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                                next(stream)
                torch.testing.assert_close(model[0].weight, torch.tensor([[2.]]))
                self.assertEqual([module.training for module in model.modules()], original_modes)
                self.assertEqual(trainer.model.stage, 'train')
                self.assertIs(trainer.current_task, original_task)
                self.assertFalse(hasattr(trainer, '_test_model_state'))

    def test_invalid_checkpoint_restores_prediction_context(self):
        trainer = self.trainer(nn.Linear(1, 1, bias=False))
        trainer.model.stage = 'val'
        original_task = object()
        trainer.current_task = original_task
        original_weight = trainer.model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.ckpt'
            torch.save({'model_state': {'missing': torch.ones(1)}}, path)
            with self.assertRaisesRegex(RuntimeError, 'state_dict'):
                self.quiet(list, trainer.predict(DataLoader(torch.ones(1, 1)), weights=path))
        torch.testing.assert_close(trainer.model.weight, original_weight)
        self.assertEqual(trainer.model.stage, 'val')
        self.assertIs(trainer.current_task, original_task)
        self.assertTrue(trainer.model.training)

    def test_test_weight_selection_keeps_optimizer_epoch_and_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckCallback('loss', on_stage='train', ckpt_dir=directory)
            other_callback = CheckCallback('loss', on_stage='val', ckpt_dir=directory)
            model = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.zero_()
            raw = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
            trainer = self.trainer(model, loss=nn.functional.mse_loss, opt=raw, callbacks=[callback, other_callback], epochs=1)
            loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)), batch_size=1)
            self.quiet(trainer.fit, loader)
            saved_weight = model.weight.detach().clone()
            path = Path(directory) / 'first-epoch.ckpt'
            save_state(trainer, trainer.model, trainer.opt, path, epoch=0)
            self.quiet(trainer.fit, loader, epochs=1)
            callback.ckpt_path = path
            if hasattr(torch.amp, 'GradScaler'):
                trainer.scaler = torch.amp.GradScaler('cpu', init_scale=512.)
            current_weight = model.weight.detach().clone()
            self.assertFalse(torch.equal(saved_weight, current_weight))
            other_path = Path(directory) / 'other-monitor.ckpt'
            other_weight = torch.full_like(current_weight, .9)
            with torch.no_grad():
                model.weight.copy_(other_weight)
            save_state(trainer, trainer.model, trainer.opt, other_path, epoch=0)
            with torch.no_grad():
                model.weight.copy_(current_weight)
            other_callback.ckpt_path = other_path
            optimizer_state = deepcopy(trainer.opt.state_dict())
            scaler_state = deepcopy(trainer.scaler.state_dict())
            epoch = trainer.init_epoch
            selections = [('current', 'current', current_weight), ('callback', other_callback, other_weight),
                          ('path', path, saved_weight), ('auto', 'auto', saved_weight)]
            for label, selection, expected_weight in selections:
                with self.subTest(weights=label):
                    result = self.quiet(trainer.test, loader, weights=selection)
                    self.assertAlmostEqual(float(result['loss']), float((expected_weight - 1).square()), places=6)
                    torch.testing.assert_close(model.weight, current_weight)
                    self.assertEqual(trainer.opt.state_dict(), optimizer_state)
                    self.assertEqual(trainer.scaler.state_dict(), scaler_state)
                    self.assertEqual(trainer.init_epoch, epoch)


if __name__ == '__main__':
    unittest.main()
