"""用 CPU/Gloo 双进程比较训练更新、全局指标和断点恢复的实际结果。"""

import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, EpochTask, Optimizer, PatchBase, Trainer
from deepepochs import metrics as dm
from deepepochs.distributed import DistributedContext


class FixedLogits(nn.Module):
    """保留 DDP 所需的可训练参数，同时让分类参考值不随训练改变。"""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))
        self.register_buffer('bias', torch.zeros(2))

    def forward(self, logits):
        return logits + self.bias + self.anchor * 0


def _classification_data():
    logits = torch.tensor([[4., 0.], [0., 4.], [4., 0.], [4., 0.], [4., 0.]])
    targets = torch.tensor([0, 1, 1, 1, 1])
    return logits, targets


def _trainer(model, loss, opt=None, **kwargs):
    return Trainer(model, loss, opt, device='cpu', precision='fp32', show_info=False,
                   log_batch=False, **kwargs)


def _regression_reference(size, global_batch_size):
    """普通单进程 PyTorch 更新，独立于 Trainer 的分片和累积实现。"""
    x = torch.arange(1., size + 1).view(-1, 1) / size
    y = 2 * x
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.25)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for start in range(0, size, global_batch_size):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(x[start:start + global_batch_size]),
                               y[start:start + global_batch_size]).backward()
        optimizer.step()
    return model.weight.detach().clone()


def _gradient_checks(context, stage):
    class CountUpdates(Callback):
        def __init__(self):
            super().__init__()
            self.count = 0

        def on_after_optimize(self, trainer, optimize_index):
            self.count += 1

    # 8 个样本：每进程批量 2 对应全局 4；累积两批对应全局 8。
    # 10 个样本覆盖不足两批的末窗口；6 个样本覆盖窗口内 2/1 的不等批量。
    cases = [(8, 1, False), (8, 2, False), (10, 2, False), (6, 2, False), (8, 1, True)]
    for size, accumulation, split_batches in cases:
        stage(f'gradient:n={size},accumulation={accumulation},split={split_batches}')
        x = torch.arange(1., size + 1).view(-1, 1) / size
        dataset = TensorDataset(x, 2 * x)
        loader_batch_size = 4 if split_batches else 2
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.25)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        updates = CountUpdates()
        trainer = _trainer(model, nn.functional.mse_loss, optimizer, epochs=1,
                           split_batches=split_batches,
                           gradient_accumulation_steps=accumulation, callbacks=[updates])
        try:
            trainer.fit(DataLoader(dataset, batch_size=loader_batch_size))
            actual = context.unwrap_model(trainer.model.model).weight.detach()
            torch.testing.assert_close(actual, _regression_reference(size, 4 * accumulation),
                                       rtol=1e-6, atol=1e-7)
            expected_updates = (size + 4 * accumulation - 1) // (4 * accumulation)
            assert updates.count == expected_updates, (updates.count, expected_updates)
        finally:
            trainer.close()


def _metric_checks(context, stage):
    logits, targets = _classification_data()
    dataset = TensorDataset(logits, targets)
    expected_loss = nn.functional.cross_entropy(logits, targets).item()

    # 函数定义在子进程内部；分布式聚合不得 pickle 指标函数。
    def tensor_f1(preds, labels):
        return dm.f1(preds, labels, average='macro')

    def confusion_f1(matrix):
        return dm.f1(conf_mat=matrix, average='macro')

    for patch_type, metric in [('tensor', tensor_f1), ('confusion', confusion_f1)]:
        stage(f'global-metrics:{patch_type}')
        metric.__name__ = 'macro_f1'
        model = FixedLogits()
        trainer = _trainer(model, nn.functional.cross_entropy, epochs=1,
                           metrics=[metric], metric_patch=patch_type)
        try:
            # 验证开始需要统一 buffer；rank 1 的偏置否则会改变损失。
            model.bias.copy_(torch.tensor([float(context.rank), -float(context.rank)]))
            result = trainer.test(DataLoader(dataset, batch_size=2))
            assert abs(float(result['loss']) - expected_loss) < 1e-6, result
            # 全局混淆矩阵为 [[1,0],[3,1]]，macro-F1=0.4；rank 指标平均不同。
            assert abs(float(result['macro_f1']) - 0.4) < 1e-6, result

            stage(f'empty-validation-rank:{patch_type}')
            tiny = TensorDataset(logits[:1], targets[:1])
            tiny_result = trainer.test(DataLoader(tiny, batch_size=2))
            assert abs(float(tiny_result['loss']) -
                       nn.functional.cross_entropy(logits[:1], targets[:1]).item()) < 1e-6
            assert abs(float(tiny_result['macro_f1']) - 0.5) < 1e-6, tiny_result
        finally:
            trainer.close()

    stage('multiple-tasks')

    def first_f1(preds, labels):
        return dm.f1(preds, labels, average='macro')

    def second_accuracy(preds, labels):
        return (preds.argmax(1) == labels).float().mean()

    trainer = _trainer(FixedLogits(), nn.functional.cross_entropy, epochs=1)
    try:
        tasks = [EpochTask(DataLoader(dataset, batch_size=2), metrics=[first_f1], name='full'),
                 EpochTask(DataLoader(TensorDataset(logits[:1], targets[:1]), batch_size=2),
                           metrics=[second_accuracy], name='tiny')]
        result = trainer.test(tasks=tasks)
        assert abs(float(result['full/first_f1']) - 0.4) < 1e-6, result
        assert float(result['tiny/second_accuracy']) == 1., result
        assert abs(float(result['full/loss']) - expected_loss) < 1e-6, result
        assert abs(float(result['tiny/loss']) -
                   nn.functional.cross_entropy(logits[:1], targets[:1]).item()) < 1e-6, result
        assert 'loss' not in result, result
    finally:
        trainer.close()

    stage('local-patch-class-and-function')

    def ratio(correct, count):
        return correct / count

    class LocalAccuracyPatch(PatchBase):
        def __init__(self):
            super().__init__()
            self.metric = ratio

        def load(self, preds, labels):
            self.correct = (preds.argmax(1) == labels).sum().detach()
            self.count = len(labels)
            return self

        def forward(self):
            return self.metric(self.correct, self.count)

        def add(self, other):
            self.correct = self.correct + other.correct
            self.count += other.count
            return self

    class LocalTask(EpochTask):
        def step(self, batch_x, batch_y, **step_args):
            output = self.model(batch_x)
            self.loss(output, batch_y)
            return {'custom_accuracy': LocalAccuracyPatch().load(output, batch_y)}

    trainer = _trainer(FixedLogits(), nn.functional.cross_entropy, epochs=1)
    try:
        result = trainer.test(tasks=LocalTask(DataLoader(dataset, batch_size=2)))
        assert abs(float(result['custom_accuracy']) - 0.4) < 1e-6, result
        assert abs(float(result['loss']) - expected_loss) < 1e-6, result
    finally:
        trainer.close()


def _step_scheduler_checks(context, stage):
    # 第一组的两个 rank 在第二步分别变好/变坏；使用局部 loss 会调出不同 LR。
    # 第二组最后一小批 loss 很小、整个累积窗口 loss 很大，不能只用末批 loss。
    for labels, accumulation in [([1., 1., 0., 10.], 1),
                                  ([1., 1., 1., 1., 10., 10., 0., 0.], 2)]:
        stage(f'step-loss-scheduler:accumulation={accumulation}')
        targets = torch.tensor(labels).view(-1, 1)
        inputs = torch.ones_like(targets)

        reference = nn.Linear(1, 1, bias=False)
        reference.weight.data.zero_()
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        reference_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            reference_optimizer, patience=0, factor=0.5)
        global_batch = 2 * accumulation
        for start in range(0, len(labels), global_batch):
            reference_optimizer.zero_grad()
            loss = nn.functional.mse_loss(reference(inputs[start:start + global_batch]),
                                          targets[start:start + global_batch])
            loss.backward()
            reference_optimizer.step()
            reference_scheduler.step(loss.item())

        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)
        trainer = _trainer(model, nn.functional.mse_loss,
                           Optimizer(optimizer, scheduler, sched_on='step', sched_with_loss=True),
                           epochs=1, gradient_accumulation_steps=accumulation)
        try:
            trainer.fit(DataLoader(TensorDataset(inputs, targets), batch_size=1))
            torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-7)
            expected_lr = reference_optimizer.param_groups[0]['lr']
            assert expected_lr == 0.05
            assert optimizer.param_groups[0]['lr'] == expected_lr, optimizer.param_groups
            learning_rates = context.all_gather_object(optimizer.param_groups[0]['lr'])
            assert learning_rates == [expected_lr, expected_lr], learning_rates
        finally:
            trainer.close()


def _checkpoint_checks(context, directory, stage):
    stage('early-stop-and-checkpoint')
    logits, targets = _classification_data()
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    model = FixedLogits()
    raw_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(raw_optimizer, step_size=1, gamma=0.5)
    optimizer = Optimizer(raw_optimizer, scheduler)
    checker = CheckCallback('loss', on_stage='val', patience=2, save_best=False,
                            ckpt_dir=directory)
    trainer = _trainer(model, nn.functional.cross_entropy, optimizer, epochs=8,
                       callbacks=[checker], running_id='shared')
    try:
        save = torch.save
        saved_paths = []

        def record_save(*args, **kwargs):
            destination = args[1] if len(args) > 1 else kwargs.get('f')
            if isinstance(destination, (str, os.PathLike)):
                saved_paths.append(destination)
            return save(*args, **kwargs)

        with patch('torch.save', side_effect=record_save):
            progress = trainer.fit(loader, loader)
        assert bool(saved_paths) == context.is_main_process, saved_paths
        assert len(progress['train']['loss']) == 3, progress
        assert trainer.init_epoch == 3, trainer.init_epoch
        assert checker.worse_times == 2, checker.worse_times
        assert abs(raw_optimizer.param_groups[0]['lr'] - 0.00625) < 1e-12
        states = context.all_gather_object((trainer.init_epoch, checker.worse_times,
                                           raw_optimizer.param_groups[0]['lr']))
        assert states == [(3, 2, 0.00625)] * 2, states
        path = Path(directory) / 'shared' / 'checkpoint.ckpt'
        assert path.is_file(), path
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert set(checkpoint['model_state']) == {'anchor', 'bias'}, checkpoint['model_state']
    finally:
        trainer.close()

    stage('distributed-checkpoint-restore')
    restored = FixedLogits()
    restored.anchor.data.fill_(7.)
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.9)
    restored_sched = torch.optim.lr_scheduler.StepLR(restored_opt, step_size=1, gamma=0.5)
    restored_check = CheckCallback('loss', on_stage='val', patience=2, save_best=False,
                                   ckpt_dir=directory)
    resumed = _trainer(restored, nn.functional.cross_entropy, Optimizer(restored_opt, restored_sched),
                       epochs=8, callbacks=[restored_check])
    try:
        resumed.load_checkpoint(str(path))
        assert resumed.init_epoch == 3
        assert restored.anchor.item() == 0.
        assert restored_sched.last_epoch == 3
        assert abs(restored_opt.param_groups[0]['lr'] - 0.00625) < 1e-12
        assert restored_check.worse_times == 2
        assert abs(float(restored_check.best_value) -
                   nn.functional.cross_entropy(logits, targets).item()) < 1e-6
    finally:
        resumed.close()


def _worker(rank, port, directory, case):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='2',
                      RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)

    def stage(message):
        (Path(directory) / f'rank-{rank}.stage').write_text(message, encoding='utf-8')

    stage('initialize-gloo')
    context = DistributedContext('cpu')
    try:
        if case == 'gradients':
            _gradient_checks(context, stage)
        elif case == 'metrics':
            _metric_checks(context, stage)
        elif case == 'checkpoint':
            _checkpoint_checks(context, directory, stage)
        elif case == 'step-scheduler':
            _step_scheduler_checks(context, stage)
        elif case in ('named-monitors', 'prediction', 'stochastic-resume'):
            from multigpu_evaluation import (
                named_dataset_monitors_case, stochastic_epoch_resume_case,
                unlabeled_prediction_case,
            )
            cases = {'named-monitors': named_dataset_monitors_case,
                     'prediction': unlabeled_prediction_case,
                     'stochastic-resume': stochastic_epoch_resume_case}
            stage(case)
            cases[case](context, 'fp32', Path(directory))
        else:
            raise ValueError(case)
        stage('complete')
        context.barrier()
    finally:
        context.close()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要 Gloo')
class DistributedTrainingTests(unittest.TestCase):
    def _run_case(self, case, directory):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        workers = mp.spawn(_worker, args=(port, directory, case), nprocs=2, join=False)
        deadline = time.monotonic() + 60
        try:
            # 子进程异常由 spawn 的 join 立即带 traceback 抛回；只有通信挂起
            # 才使用上限，并附上各进程最后进入的阶段，便于定位不同步位置。
            while not workers.join(timeout=1):
                if time.monotonic() >= deadline:
                    stages = {path.name: path.read_text(encoding='utf-8')
                              for path in Path(directory).glob('rank-*.stage')}
                    self.fail(f'{case} 双进程在 60 秒内未完成：{stages}')
        finally:
            for process in workers.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)

    def test_global_gradients_accumulation_and_split_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('gradients', directory)

    def test_global_metrics_uneven_validation_and_custom_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('metrics', directory)

    def test_step_loss_scheduler_uses_global_accumulation_window(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('step-scheduler', directory)

    def test_named_dataset_losses_and_explicit_epoch_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('named-monitors', directory)

    def test_unlabeled_prediction_has_exact_sample_ids_and_restores_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('prediction', directory)

    def test_stochastic_epoch_resume_matches_continuous_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('stochastic-resume', directory)

    def test_early_stopping_scheduler_and_portable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case('checkpoint', directory)
            # 子进程已退出；同一 checkpoint 应可直接恢复到普通单进程模型。
            model = FixedLogits()
            model.anchor.data.fill_(7.)
            raw_optimizer = torch.optim.SGD(model.parameters(), lr=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(raw_optimizer, step_size=1, gamma=0.5)
            checker = CheckCallback('loss', on_stage='val', patience=2, save_best=False,
                                    ckpt_dir=directory)
            trainer = _trainer(model, nn.functional.cross_entropy, Optimizer(raw_optimizer, scheduler),
                               epochs=8, callbacks=[checker])
            try:
                trainer.load_checkpoint(str(Path(directory) / 'shared' / 'checkpoint.ckpt'))
                self.assertFalse(trainer.distributed.is_distributed)
                self.assertEqual(model.anchor.item(), 0.)
                self.assertEqual(trainer.init_epoch, 3)
                self.assertEqual(scheduler.last_epoch, 3)
                self.assertAlmostEqual(raw_optimizer.param_groups[0]['lr'], 0.00625)
                self.assertEqual(checker.worse_times, 2)
            finally:
                trainer.close()


if __name__ == '__main__':
    unittest.main()
