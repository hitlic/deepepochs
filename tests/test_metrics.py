import unittest

import torch

from deepepochs import metrics


class MetricsTests(unittest.TestCase):
    def test_binary_averages_include_both_classes(self):
        matrix = torch.tensor([[5, 2], [3, 10]])
        expected = {
            'micro': (0.75, 0.75, 0.75),
            'macro': ((5 / 8 + 10 / 12) / 2, (5 / 7 + 10 / 13) / 2,
                      (10 / 15 + 20 / 25) / 2),
            'weighted': ((5 / 8 * 7 + 10 / 12 * 13) / 20,
                         0.75, (10 / 15 * 7 + 20 / 25 * 13) / 20),
        }
        for average, values in expected.items():
            for metric, value in zip((metrics.precision, metrics.recall, metrics.f1), values):
                with self.subTest(average=average, metric=metric.__name__):
                    actual = metric(conf_mat=matrix, average=average)
                    self.assertAlmostEqual(float(actual), value, places=6)

    def test_perfect_binary_macro_is_one(self):
        matrix = torch.tensor([[3, 0], [0, 5]])
        for metric in (metrics.precision, metrics.recall, metrics.f1):
            self.assertEqual(float(metric(conf_mat=matrix, average='macro')), 1.0)

    def test_metrics_follow_in_place_confusion_matrix_updates(self):
        matrix = torch.eye(2, dtype=torch.int64)
        self.assertEqual(float(metrics.precision(conf_mat=matrix, average='macro')), 1.)
        matrix[1, 0] = 3
        expected = {
            'micro': (0.4, 0.4, 0.4),
            'macro': (0.625, 0.625, 0.4),
            'weighted': (0.85, 0.4, 0.4),
        }
        for average, values in expected.items():
            for metric, value in zip((metrics.precision, metrics.recall, metrics.f1), values):
                with self.subTest(average=average, metric=metric.__name__):
                    self.assertAlmostEqual(float(metric(conf_mat=matrix, average=average)), value, places=6)

    def test_accuracy_follows_reused_prediction_buffer(self):
        predictions = torch.tensor([[1., 0.], [0., 1.]])
        targets = torch.tensor([0, 1])
        self.assertEqual(float(metrics.accuracy(predictions, targets)), 1.)
        predictions.copy_(predictions.flip(1))
        self.assertEqual(float(metrics.accuracy(predictions, targets)), 0.)
        torch.testing.assert_close(metrics.confusion_matrix(predictions, targets, 2),
                                   torch.tensor([[0, 1], [1, 0]], dtype=torch.int32))

    def test_accuracy_follows_reused_target_buffer(self):
        predictions = torch.tensor([[1., 0.], [0., 1.]])
        targets = torch.tensor([0, 1])
        self.assertEqual(float(metrics.accuracy(predictions, targets)), 1.)
        targets.copy_(targets.flip(0))
        self.assertEqual(float(metrics.accuracy(predictions, targets)), 0.)

    def test_single_output_binary_does_not_infer_classes_from_batch_labels(self):
        targets = torch.tensor([0, 0])
        for preds in (torch.tensor([0.1, 0.9]), torch.tensor([[0.1], [0.9]])):
            self.assertEqual(metrics.get_class_num(preds, targets), 2)
            self.assertEqual(float(metrics.accuracy(preds, targets)), 0.5)


if __name__ == '__main__':
    unittest.main()
