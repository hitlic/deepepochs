import unittest

import torch

from deepepochs.loops import TensorDict, TensorTuple
from deepepochs.tools import groupby_apply


class NestedDataTests(unittest.TestCase):
    def test_nested_label_mask_moves_with_outer_tuple(self):
        labels = TensorTuple((torch.tensor([1, 0]), (torch.tensor([True, False]),)))
        moved = labels.to('meta')
        self.assertIsInstance(moved, TensorTuple)
        self.assertIsInstance(moved[1], TensorTuple)
        self.assertEqual(moved[0].device.type, 'meta')
        self.assertEqual(moved[1][0].device.type, 'meta')
        self.assertEqual(labels[1][0].device.type, 'cpu')

    def test_mixed_nested_containers_clone_and_detach_all_tensors(self):
        first = torch.tensor([1.], requires_grad=True)
        inner = torch.tensor([2.], requires_grad=True)
        data = TensorDict({'inputs': (first, {'mask': [inner, None]}), 'name': 'sample'})
        cloned = data.clone()
        self.assertIsInstance(cloned, TensorDict)
        self.assertIsInstance(cloned['inputs'], TensorTuple)
        self.assertIsInstance(cloned['inputs'][1], TensorDict)
        self.assertNotEqual(cloned['inputs'][0].data_ptr(), first.data_ptr())
        self.assertNotEqual(cloned['inputs'][1]['mask'][0].data_ptr(), inner.data_ptr())
        self.assertEqual(cloned['name'], 'sample')
        self.assertIsNone(cloned['inputs'][1]['mask'][1])
        for detached in (data.detach(), data.data):
            self.assertFalse(detached['inputs'][0].requires_grad)
            self.assertFalse(detached['inputs'][1]['mask'][0].requires_grad)
            self.assertIsNone(detached['inputs'][1]['mask'][1])
        on_cpu = data.cpu()
        self.assertIsInstance(on_cpu['inputs'][1]['mask'], TensorTuple)
        self.assertEqual(on_cpu['inputs'][1]['mask'][0].device.type, 'cpu')

    def test_nested_dtype_operations_reach_every_tensor(self):
        data = TensorTuple((torch.tensor([1.5]), ({'value': [torch.tensor([2.5])]},)))
        conversions = [
            (data.to('cpu', dtype=torch.float64), torch.float64),
            (data.float(), torch.float32),
            (data.long(), torch.int64),
            (data.int(), torch.int32),
        ]
        for converted, dtype in conversions:
            with self.subTest(dtype=dtype):
                self.assertEqual(converted[0].dtype, dtype)
                self.assertEqual(converted[1][0]['value'][0].dtype, dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_nested_label_mask_round_trip_cuda_cpu(self):
        labels = TensorTuple((torch.tensor([1, 0]), (torch.tensor([True, False]),)))
        moved = labels.to('cuda')
        self.assertEqual(moved[0].device.type, 'cuda')
        self.assertEqual(moved[1][0].device.type, 'cuda')
        restored = moved.cpu()
        torch.testing.assert_close(restored[0], labels[0])
        torch.testing.assert_close(restored[1][0], labels[1][0])


class GroupByTests(unittest.TestCase):
    def test_unsorted_keys_group_values_before_mean_and_sum(self):
        values = torch.tensor([[3., 30.], [2., 20.], [5., 50.], [4., 40.]])
        keys = torch.tensor([1, 0, 1, 0])
        torch.testing.assert_close(groupby_apply(values, keys, 'mean'), torch.tensor([[3., 30.], [4., 40.]]))
        torch.testing.assert_close(groupby_apply(values, keys, 'sum'), torch.tensor([[6., 60.], [8., 80.]]))

    def test_sorted_keys_keep_ascending_group_output(self):
        values = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]])
        keys = torch.tensor([0, 0, 1, 1, 1])
        torch.testing.assert_close(groupby_apply(values, keys), torch.tensor([[1.5, 1.5], [4., 4.]]))

    def test_grouped_mean_preserves_gradients(self):
        values = torch.tensor([2., 4., 6.], requires_grad=True)
        result = groupby_apply(values, torch.tensor([2, -1, 2]))
        torch.testing.assert_close(result, torch.tensor([4., 4.]))
        result.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([0.5, 1., 0.5]))


if __name__ == '__main__':
    unittest.main()
