from io import BytesIO
from types import SimpleNamespace
import unittest

import torch

from deepepochs.callbacks import InterpreteCallback
from deepepochs.loops import TensorDict, TensorTuple


class InterpreteInputTests(unittest.TestCase):
    def setUp(self):
        self.preds = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])
        self.targets = torch.tensor([1, 0, 1])

    def callback(self, **kwargs):
        return InterpreteCallback(metric=lambda preds, targets: preds[:, 1], k=3, **kwargs)

    def test_single_input_keeps_tensor_feature_structure(self):
        inputs = torch.arange(6.).reshape(3, 2).requires_grad_()
        callback = self.callback()
        callback.put_queue(self.preds, self.targets, inputs)
        self.assertEqual(len(callback.top_samples()), 3)
        for sample, index in zip(callback.top_samples(), [0, 2, 1]):
            self.assertIsInstance(sample[3], torch.Tensor)
            torch.testing.assert_close(sample[3], inputs[index])
            self.assertFalse(sample[3].requires_grad)
            self.assertEqual(sample[3].device.type, 'cpu')

    def test_two_inputs_preserve_all_three_samples_and_both_features(self):
        inputs = TensorTuple((torch.arange(6.).reshape(3, 2), torch.arange(9.).reshape(3, 3) + 10))
        callback = self.callback()
        callback.put_queue(self.preds, self.targets, inputs)
        self.assertEqual(len(callback.top_samples()), 3)
        for sample, index in zip(callback.top_samples(), [0, 2, 1]):
            self.assertIsInstance(sample[3], tuple)
            self.assertEqual(len(sample[3]), 2)
            torch.testing.assert_close(sample[3][0], inputs[0][index])
            torch.testing.assert_close(sample[3][1], inputs[1][index])

    def test_dictionary_features_are_sliced_by_sample(self):
        inputs = TensorDict({'x': torch.arange(6.).reshape(3, 2),
                             'extra': {'mask': torch.tensor([[True], [False], [True]])}})
        callback = self.callback()
        callback.put_queue(self.preds, self.targets, inputs)
        for sample, index in zip(callback.top_samples(), [0, 2, 1]):
            self.assertIsInstance(sample[3], dict)
            torch.testing.assert_close(sample[3]['x'], inputs['x'][index])
            torch.testing.assert_close(sample[3]['extra']['mask'], inputs['extra']['mask'][index])

    def test_multi_input_image_logging_uses_first_input_tensor(self):
        images = torch.arange(12.).reshape(3, 1, 2, 2)
        extra = torch.arange(6.).reshape(3, 2)
        for inputs in ((images, extra), {'image': images, 'extra': extra}):
            with self.subTest(container=type(inputs).__name__):
                logged = []
                callback = self.callback(image_data=True, stages='test')
                callback.logger = SimpleNamespace(add_image=lambda tag, image, index: logged.append(image))
                callback.put_queue(self.preds, self.targets, inputs)
                callback.on_after_test_epochs(SimpleNamespace(main_process=True), [], {})
                self.assertEqual(len(logged), 3)
                for image, index in zip(logged, [0, 2, 1]):
                    self.assertIsInstance(image, torch.Tensor)
                    torch.testing.assert_close(image, images[index])

    def test_top_sample_owns_only_one_image_prediction_and_target(self):
        inputs = torch.arange(32 * 3 * 64 * 64.).reshape(32, 3, 64, 64).requires_grad_()
        preds = torch.arange(32 * 1000.).reshape(32, 1000).requires_grad_()
        targets = torch.arange(32)
        callback = InterpreteCallback(metric=lambda output, labels: output[:, 0], k=1)
        callback.put_queue(preds, targets, inputs)
        sample = callback.top_samples()[0]

        # 一张图为 49,152 bytes，不能因为 CPU 切片而保留整批 1,572,864 bytes。
        self.assertEqual(sample[3].untyped_storage().nbytes(), 49_152)
        for retained, original in ((sample[1], preds), (sample[2], targets), (sample[3], inputs)):
            self.assertEqual(retained.untyped_storage().nbytes(), retained.numel() * retained.element_size())
            self.assertNotEqual(retained.untyped_storage().data_ptr(), original.untyped_storage().data_ptr())
            self.assertEqual(retained.device.type, 'cpu')
            self.assertFalse(retained.requires_grad)
            torch.testing.assert_close(retained, original[31])

        serialized = BytesIO()
        torch.save(callback.top_queue.items(), serialized)
        tensor_bytes = sum(value.numel() * value.element_size() for value in sample[1:])
        self.assertLess(len(serialized.getvalue()), tensor_bytes + 8192)
        self.assertLess(len(serialized.getvalue()), inputs.untyped_storage().nbytes() // 10)

        with torch.no_grad():
            inputs.zero_()
            preds.zero_()
            targets.zero_()
        self.assertEqual(int(sample[2]), 31)
        self.assertEqual(float(sample[1][0]), 31_000.)
        self.assertEqual(float(sample[3].flatten()[0]), 31 * 3 * 64 * 64)

    def test_nested_multi_input_top_sample_copies_each_tensor_leaf(self):
        images = torch.ones(32, 3, 64, 64, requires_grad=True)
        other_images = torch.full((32, 3, 64, 64), 2., requires_grad=True)
        features = torch.arange(32 * 8.).reshape(32, 8).requires_grad_()
        inputs = (images, {'extra': [other_images, features]})
        preds = torch.arange(32 * 2.).reshape(32, 2).requires_grad_()
        targets = torch.arange(32)
        callback = InterpreteCallback(metric=lambda output, labels: output[:, 0], k=1)
        callback.put_queue(preds, targets, inputs)
        sample = callback.top_samples()[0]
        retained = (sample[1], sample[2], sample[3][0], sample[3][1]['extra'][0],
                    sample[3][1]['extra'][1])
        originals = (preds, targets, images, other_images, features)
        for value, original in zip(retained, originals):
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
            self.assertNotEqual(value.untyped_storage().data_ptr(), original.untyped_storage().data_ptr())
            self.assertEqual(value.device.type, 'cpu')
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, original[31])
        serialized = BytesIO()
        torch.save(callback.top_queue.items(), serialized)
        tensor_bytes = sum(value.numel() * value.element_size() for value in retained)
        self.assertLess(len(serialized.getvalue()), tensor_bytes + 8192)


if __name__ == '__main__':
    unittest.main()
