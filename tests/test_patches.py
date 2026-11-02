import pickle
import unittest
import weakref

import torch

from deepepochs import TrainerBase, metrics
from deepepochs.patches import (
    ConfusionMetrics, ConfusionPatch, MeanPatch, PatchBase, TensorPatch,
    ValuePatch, aggregate_patch_dicts, run_patch_dicts,
)


class PatchesTests(unittest.TestCase):
    def test_unlabeled_tensor_patch_concatenates_predictions(self):
        def metric(preds, targets):
            self.assertIsNone(targets)
            return preds.mean()

        first = TensorPatch(metric).load(torch.tensor([[1.], [2.]]), None)
        second = TensorPatch(metric).load(torch.tensor([[3.], [4.], [5.]]), None)
        self.assertEqual(float(first()), 1.5)
        self.assertEqual(float((first + second)()), 3.0)
        self.assertEqual(float(first()), 1.5)

    def test_nested_dictionary_outputs_are_concatenated(self):
        def metric(preds, targets):
            self.assertIsNone(targets)
            self.assertEqual(preds['nested']['aux'].shape, (3, 1))
            self.assertEqual(preds['loss'].shape, (2,))
            return preds['logits'].mean()

        first = {'logits': torch.tensor([[1.], [2.]]),
                 'nested': {'aux': torch.zeros(2, 1)}, 'loss': torch.tensor(3.)}
        second = {'logits': torch.tensor([[6.]]),
                  'nested': {'aux': torch.ones(1, 1)}, 'loss': torch.tensor(5.)}
        result = TensorPatch(metric).load(first, None) + TensorPatch(metric).load(second, None)
        self.assertEqual(float(result()), 3.0)

    def test_tensor_patch_detaches_nested_predictions_and_targets(self):
        def metric(preds, targets):
            for data in (preds, targets):
                self.assertIsInstance(data['nested'], tuple)
                self.assertIsInstance(data['nested'][1], list)
                self.assertIsNone(data['nested'][1][1])
            return (preds['value'] - targets['value']).abs().mean()

        patches = []
        originals = []
        for values in ([1., 3.], [5.]):
            inputs = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            labels = torch.zeros_like(inputs, requires_grad=True)
            preds = {'value': inputs * 2, 'nested': (inputs.sin(), [inputs.cos(), None])}
            targets = {'value': labels * 3, 'nested': (labels.square(), [labels.exp(), None])}
            originals.append((inputs, labels, preds, targets))
            patch = TensorPatch(metric).load(preds, targets)
            patches.append(patch)
            for stored, original in ((patch.batch_preds[0][0], preds),
                                     (patch.batch_targets[0][0], targets)):
                self.assertIsInstance(stored, dict)
                self.assertIsInstance(stored['nested'], tuple)
                self.assertIsInstance(stored['nested'][1], list)
                self.assertIsNone(stored['nested'][1][1])
                for value, source in ((stored['value'], original['value']),
                                      (stored['nested'][0], original['nested'][0]),
                                      (stored['nested'][1][0], original['nested'][1][0])):
                    self.assertFalse(value.requires_grad)
                    self.assertIsNone(value.grad_fn)
                    self.assertEqual(value.dtype, torch.float64)
                    self.assertEqual(value.data_ptr(), source.data_ptr())
                    torch.testing.assert_close(value, source)
        result = sum(patches)()
        self.assertEqual(float(result), 6.)
        self.assertFalse(result.requires_grad)
        self.assertEqual([float(patch()) for patch in patches], [4., 10.])
        for inputs, labels, preds, targets in originals:
            (preds['value'].sum() + targets['value'].sum()).backward()
            torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.))
            torch.testing.assert_close(labels.grad, torch.full_like(labels, 3.))

    def test_custom_tensor_patch_releases_unused_auxiliary_activations(self):
        activations = []
        retained_patches = []
        testcase = self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, inputs):
                logits = inputs * self.weight
                auxiliary = logits.sin()
                activations.append(weakref.ref(auxiliary))
                return {'logits': logits, 'nested': {'aux': auxiliary.square()}}

        def metric(preds, targets):
            testcase.assertFalse(preds['nested']['aux'].requires_grad)
            return (preds['logits'] - targets).abs().mean()

        class Trainer(TrainerBase):
            def train_step(self, batch_x, batch_y, **step_args):
                testcase.assertTrue(all(ref() is None for ref in activations))
                predictions = self.model(batch_x)
                self.loss(predictions['logits'], batch_y)
                patch = TensorPatch(metric).load(predictions, batch_y)
                retained_patches.append(patch)
                return {'mae': patch}

        inputs = torch.tensor([[1.], [3.], [5.]])
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=2)
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        trainer = Trainer(model, torch.nn.functional.mse_loss, optimizer, epochs=1,
                          device='cpu', log_batch=False, show_info=False)
        try:
            progress = trainer.fit(dataloader)
            self.assertEqual(float(progress['train']['mae'][0]), 3.)
            self.assertEqual(len(retained_patches), 2)
            self.assertTrue(all(ref() is None for ref in activations))
            self.assertEqual([float(patch()) for patch in retained_patches], [2., 5.])
        finally:
            trainer.close()

    def test_single_output_confusion_patches_keep_two_classes(self):
        metric = lambda matrix: metrics.accuracy(conf_mat=matrix)
        first = ConfusionPatch(metric).load(torch.tensor([0.1, 0.9]), torch.tensor([0, 0]))
        second = ConfusionPatch(metric).load(torch.tensor([[0.2], [0.8]]), torch.tensor([0, 1]))
        self.assertEqual(float((first + second)()), 0.75)
        combined = ConfusionMetrics(metrics='accuracy').load(
            torch.tensor([0.1, 0.9]), torch.tensor([0, 0]))
        self.assertEqual(float(combined()['acc']), 0.5)

    def test_shared_batch_size_and_original_patches_are_unchanged(self):
        count = torch.tensor(2)
        first = ValuePatch(count).load(torch.tensor(1.))
        second = ValuePatch(count).load(torch.tensor(3.))
        other_metric = ValuePatch(count).load(torch.tensor(7.))
        combined = first + second
        self.assertEqual(float(combined()), 2.)
        self.assertEqual(int(count), 2)
        self.assertEqual(float(first()), 1.)
        self.assertEqual(float(second()), 3.)
        self.assertEqual(float(other_metric()), 7.)

    def test_mean_patch_detaches_differentiable_metric_statistics(self):
        for dictionary in (False, True):
            with self.subTest(dictionary=dictionary):
                source_values = []

                def metric(preds, targets):
                    value = (preds - targets).abs().mean()
                    source_values.append((preds, value))
                    return {'mae': value, 'constant': 1.5} if dictionary else value

                patches = []
                for values in ([1., 3.], [5.]):
                    preds = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                    patches.append(MeanPatch(metric, len(values)).load(preds, torch.zeros_like(preds)))
                result = sum(patches)()
                value = result['mae'] if dictionary else result
                self.assertEqual(float(value), 3.)
                self.assertEqual(value.dtype, torch.float64)
                self.assertEqual(value.device.type, 'cpu')
                self.assertFalse(value.requires_grad)
                for patch, expected in zip(patches, (2., 5.)):
                    original = patch()['mae'] if dictionary else patch()
                    self.assertEqual(float(original), expected)
                    stored = patch.batch_value['mae'] if dictionary else patch.batch_value
                    self.assertFalse(stored.requires_grad)
                sum(value for _, value in source_values).backward()
                for inputs, _ in source_values:
                    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 1 / inputs.numel()))
                if dictionary:
                    self.assertEqual(result['constant'], 1.5)
                    self.assertIsInstance(result['constant'], float)

    def test_value_patch_detaches_differentiable_scalar_statistics(self):
        for dictionary in (False, True):
            with self.subTest(dictionary=dictionary):
                patches = []
                source_values = []
                for values in ([1., 3.], [5.]):
                    inputs = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                    mean = inputs.abs().mean()
                    source_values.append((inputs, mean))
                    value = {'mae': mean, 'constant': 1.5} if dictionary else mean
                    patches.append(ValuePatch(len(values)).load(value))
                    self.assertTrue(mean.requires_grad)
                result = sum(patches)()
                value = result['mae'] if dictionary else result
                self.assertEqual(float(value), 3.)
                self.assertEqual(value.dtype, torch.float64)
                self.assertEqual(value.device.type, 'cpu')
                self.assertFalse(value.requires_grad)
                for patch, expected in zip(patches, (2., 5.)):
                    original = patch()['mae'] if dictionary else patch()
                    self.assertEqual(float(original), expected)
                    stored = patch.batch_value['mae'] if dictionary else patch.batch_value
                    self.assertFalse(stored.requires_grad)
                sum(value for _, value in source_values).backward()
                for inputs, _ in source_values:
                    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 1 / inputs.numel()))
                if dictionary:
                    self.assertEqual(result['constant'], 1.5)
                    self.assertIsInstance(result['constant'], float)

    def test_custom_training_step_can_aggregate_differentiable_metrics(self):
        class Trainer(TrainerBase):
            def train_step(self, batch_x, batch_y, **step_args):
                predictions = self.model(batch_x)
                self.loss(predictions, batch_y)
                metric = lambda preds, targets: (preds - targets).abs().mean()
                return {'mae': MeanPatch(metric, len(batch_y)).load(predictions, batch_y)}

        inputs = torch.tensor([[1.], [3.], [5.]])
        targets = torch.zeros_like(inputs)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets), batch_size=2)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        trainer = Trainer(model, torch.nn.functional.mse_loss, optimizer, epochs=1,
                          device='cpu', log_batch=False, show_info=False)
        try:
            progress = trainer.fit(dataloader)
            self.assertEqual(float(progress['train']['mae'][0]), 3.)
        finally:
            trainer.close()

    def test_empty_patch_dicts(self):
        self.assertEqual(run_patch_dicts([]), {})

    def test_distributed_patches_transfer_only_data(self):
        # 本地函数和本地Patch类不能pickle；它们由持有指标的进程保留。
        class HitsPatch(PatchBase):
            def __init__(self):
                super().__init__('hits')
                self.at = (1, 2)
                self.hits_count = {'@1': 0, '@2': 0}

            def load(self, counts, targets=None):
                self.hits_count = counts
                return self

            def forward(self):
                return self.hits_count

            def add(self, other):
                self.hits_count = {key: self.hits_count[key] + other.hits_count[key]
                                   for key in self.hits_count}
                return self

        class GenericValuePatch(ValuePatch):
            pass

        class GenericMeanPatch(MeanPatch):
            pass

        class GenericConfusionPatch(ConfusionPatch):
            pass

        mean = lambda preds, targets: preds.mean()
        accuracy = lambda matrix: metrics.accuracy(conf_mat=matrix)
        local = {
            'loss': GenericValuePatch(2).load(torch.tensor(1.)),
            'mean': GenericMeanPatch(mean, 2).load(torch.tensor([1., 1.]), None),
            'tensor': TensorPatch(mean).load(torch.tensor([1., 1.], requires_grad=True), None),
            'confusion': GenericConfusionPatch(accuracy).load(torch.tensor([0.1, 0.9]), torch.tensor([0, 0])),
            'custom': HitsPatch().load({'@1': 1, '@2': 2}),
        }
        remote_states = {
            'loss': {'batch_size': 1, 'batch_value': torch.tensor(7.)},
            'mean': {'batch_size': 1, 'batch_value': torch.tensor(7.)},
            'tensor': {'batch_preds': [[torch.tensor([7.])]], 'batch_targets': None},
            'confusion': {'num_classes': 2, 'confusion_matrix': torch.tensor([[0, 0], [0, 1]])},
            'custom': {'at': (1, 2), 'hits_count': {'@1': 1, '@2': 1}},
        }

        class Context:
            rank = 0
            world_size = 2
            is_distributed = True

            def all_gather_object(self, data):
                pickle.dumps(data)
                if all(value is None for value in data.values()):
                    return [data, dict.fromkeys(remote_states)]
                tensor = data['tensor']['batch_preds'][0][0]
                self_test.assertEqual(tensor.device.type, 'cpu')
                self_test.assertFalse(tensor.requires_grad)
                self_test.assertNotIn('metric', data['tensor'])
                return [data, remote_states]

            def broadcast_object(self, data, src):
                self_test.assertEqual(src, 0)
                return data

        self_test = self
        actual = aggregate_patch_dicts([local], Context())
        for key in ('loss', 'mean', 'tensor'):
            self.assertEqual(float(actual[key]), 3.)
        self.assertAlmostEqual(float(actual['confusion']), 2 / 3, places=6)
        self.assertEqual(actual['hits'], {'@1': 2, '@2': 3})
        self.assertEqual(local['custom'].hits_count, {'@1': 1, '@2': 2})

    def test_empty_distributed_rank_receives_global_result(self):
        class Context:
            rank = 1
            world_size = 2
            is_distributed = True

            calls = 0

            def all_gather_object(self, data):
                self_test.assertEqual(data, {})
                self.calls += 1
                if self.calls == 1:
                    return [{'loss': None}, {}]
                return [{'loss': {'batch_size': 1, 'batch_value': 2.}}, {}]

            def broadcast_object(self, data, src):
                self_test.assertIsNone(data)
                return ('loss', 2., None)

        self_test = self
        self.assertEqual(aggregate_patch_dicts([], Context()), {'loss': 2.})

    def test_rank_zero_can_receive_a_metric_owned_only_by_rank_one(self):
        class GenericValuePatch(ValuePatch):
            pass

        class Context:
            rank = 0
            is_distributed = True

            def all_gather_object(self, data):
                if all(value is None for value in data.values()):
                    return [data, {'remote': None}]
                return [data, {'remote': {'batch_size': 1, 'batch_value': torch.tensor(7.)}}]

            def broadcast_object(self, data, src):
                if src == 0:
                    return data
                self_test.assertIsNone(data)
                return ('remote', torch.tensor(7.), None)

        self_test = self
        actual = aggregate_patch_dicts([{'local': GenericValuePatch(1).load(torch.tensor(2.))}], Context())
        self.assertEqual(float(actual['local']), 2.)
        self.assertEqual(float(actual['remote']), 7.)

    def test_owner_exceptions_are_broadcast_as_data(self):
        class LocalMetricError(ValueError):
            pass

        def bad_metric(preds, targets):
            raise LocalMetricError('cannot calculate this metric')

        class Context:
            rank = 0
            is_distributed = True

            def all_gather_object(self, data):
                return [data, {}]

            def broadcast_object(self, data, src):
                pickle.dumps(data)
                return data

        patches = [{'bad': TensorPatch(bad_metric).load(torch.tensor([1.]), None)}]
        with self.assertRaisesRegex(RuntimeError, 'LocalMetricError.*cannot calculate this metric'):
            aggregate_patch_dicts(patches, Context())

    def test_local_sum_failure_is_shared_before_metric_calculation(self):
        class Context:
            rank = 1
            is_distributed = True

            def all_gather_object(self, data):
                self_test.assertIsInstance(data, tuple)
                pickle.dumps(data)
                return [{}, data]

        self_test = self
        patches = [{'value': ValuePatch(1).load({'a': torch.tensor(1.)})},
                   {'value': ValuePatch(1).load({'b': torch.tensor(2.)})}]
        with self.assertRaisesRegex(RuntimeError, 'rank 1.*AssertionError'):
            aggregate_patch_dicts(patches, Context())

    def test_all_ranks_empty(self):
        class Context:
            rank = 0
            is_distributed = True

            def all_gather_object(self, data):
                return [{}, {}]

            def broadcast_object(self, data, src):
                return data

        self.assertEqual(aggregate_patch_dicts([], Context()), {})


if __name__ == '__main__':
    unittest.main()
