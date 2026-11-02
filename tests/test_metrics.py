import unittest

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

    def test_multiclass_averages_and_beta_match_reference_with_missing_classes(self):
        targets = [0] * 8 + [1] * 5 + [2] * 5 + [4]
        predictions = [0] * 5 + [1] * 2 + [3] + [0] + [1] * 4 + [1] * 3 + [2] * 3
        matrix = torch.tensor([[5, 2, 0, 1, 0, 0],
                               [1, 4, 0, 0, 0, 0],
                               [0, 3, 2, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        scores = torch.nn.functional.one_hot(torch.tensor(predictions), 6).float()
        inputs = ({'conf_mat': matrix}, {'preds': scores, 'targets': torch.tensor(targets)})
        for kwargs in inputs:
            self.assertAlmostEqual(float(metrics.accuracy(**kwargs)),
                                   accuracy_score(targets, predictions), places=6)
            for average in ('micro', 'macro', 'weighted'):
                for beta in (0., 0.5, 1., 2.):
                    expected = precision_recall_fscore_support(
                        targets, predictions, labels=range(6), average=average,
                        beta=beta, zero_division=0)
                    actual = (metrics.precision(**kwargs, average=average),
                              metrics.recall(**kwargs, average=average),
                              metrics.fbeta(**kwargs, beta=beta, average=average))
                    with self.subTest(from_matrix='conf_mat' in kwargs, average=average, beta=beta):
                        for value, reference in zip(actual, expected[:3]):
                            self.assertAlmostEqual(float(value), reference, places=6)
                        if beta == 1:
                            torch.testing.assert_close(metrics.f1(**kwargs, average=average), actual[2])

    def test_cmats_and_weights_keep_per_class_dictionary_contract(self):
        matrix = torch.tensor([[5, 2, 0, 1, 0, 0],
                               [1, 4, 0, 0, 0, 0],
                               [0, 3, 2, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
        cmats, weights = metrics.cmats_and_weights(matrix)
        self.assertIsInstance(cmats, list)
        self.assertIsInstance(weights, list)
        expected = [(5, 3, 1, 10), (4, 1, 5, 9), (2, 3, 1, 13),
                    (0, 0, 1, 18), (0, 1, 0, 18), (0, 0, 0, 19)]
        self.assertEqual(len(cmats), len(expected))
        for cmat, counts in zip(cmats, expected):
            self.assertIsInstance(cmat, dict)
            self.assertEqual(tuple(int(cmat[key]) for key in ('TP', 'FN', 'FP', 'TN')), counts)
            self.assertEqual(int(cmat.TP), counts[0])
        torch.testing.assert_close(torch.stack(weights), torch.tensor([8., 5., 5., 0., 1., 0.]) / 19)

    def test_zero_denominators_preserve_empty_matrix_results(self):
        matrix = torch.zeros(4, 4, dtype=torch.int64)
        for metric in (metrics.precision, metrics.recall, metrics.f1):
            for average in ('micro', 'macro'):
                with self.subTest(metric=metric.__name__, average=average):
                    self.assertEqual(float(metric(conf_mat=matrix, average=average)), 0.)
            self.assertTrue(torch.isnan(metric(conf_mat=matrix, average='weighted')))
        self.assertTrue(torch.isnan(metrics.accuracy(conf_mat=matrix)))

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
