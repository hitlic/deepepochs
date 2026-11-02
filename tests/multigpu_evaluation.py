"""torchrun下的CUDA/NCCL、CPU/Gloo评估、回调与Checkpoint联调。"""
import argparse
import contextlib
from copy import deepcopy
import io
import json
import os
import random
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from deepepochs import AnalyzeCallback, Callback, CheckCallback, EpochTask, InterpreteCallback, LogCallback, LRFindCallback, Optimizer, PatchBase, Trainer, __version__
from deepepochs import metrics
from deepepochs.callbacks.check import finalize_checkpoints
from deepepochs.distributed import DistributedContext
from deepepochs.loops import StopLoopException
from deepepochs.patches import ValuePatch, aggregate_patch_dicts


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.))
        self.register_buffer('bias', torch.zeros(2))

    def forward(self, logits):
        return logits + self.bias + self.anchor * 0


class ExactIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def classification_data(size):
    indices = torch.arange(size)
    margins = 0.5 + indices.float() / 2
    predicted = indices.remainder(3).eq(1).long()
    logits = torch.zeros(size, 2)
    logits[indices, predicted] = margins
    targets = indices.remainder(2).long()
    return logits, targets


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def make_trainer(context, precision, model=None, loss=F.cross_entropy, opt=None, **kwargs):
    # 顶层context持有进程组；Trainer新建的context借用同一组，close不会销毁它。
    return Trainer(FixedLogits() if model is None else model, loss, opt,
                   device=context.device, precision=precision, show_info=False,
                   log_batch=False, **kwargs)


def assert_close_number(actual, expected, tolerance=1e-6):
    assert abs(float(actual) - float(expected)) <= tolerance, (float(actual), float(expected))


def same_on_all_ranks(context, value):
    values = context.all_gather_object(value)
    assert all(item == values[0] for item in values), values


def weighted_metrics(context, kind):
    weights = torch.tensor([1., 3.], device=context.device)
    if kind == 'tensor':
        def metric(preds, labels):
            assert preds.device == weights.device and labels.device == weights.device
            sample_weights = weights[labels]
            return {'macro_f1': metrics.f1(preds, labels, average='macro'),
                    'weighted_accuracy': ((preds.argmax(1) == labels) * sample_weights).sum() / sample_weights.sum()}
    else:
        def metric(matrix):
            assert matrix.device == weights.device
            return {'macro_f1': metrics.f1(conf_mat=matrix, average='macro'),
                    'weighted_accuracy': (matrix.diag() * weights).sum() / (matrix.sum(1) * weights).sum()}
    metric.__name__ = ''
    return metric


def global_metrics_case(context, precision):
    size = context.world_size * 2 + 1
    logits, targets = classification_data(size)
    checked = []
    for kind in ('tensor', 'confusion'):
        trainer = make_trainer(context, precision, metrics=[weighted_metrics(context, kind)], metric_patch=kind)
        try:
            for count in (size, 1):
                inputs, labels = logits[:count].to(context.device), targets[:count].to(context.device)
                expected_loss = F.cross_entropy(inputs, labels)
                expected = weighted_metrics(context, 'tensor')(inputs, labels)
                result = quiet(trainer.test, DataLoader(TensorDataset(logits[:count], targets[:count]), batch_size=2))
                assert_close_number(result['loss'], expected_loss)
                for name in expected:
                    assert_close_number(result[name], expected[name])
                same_on_all_ranks(context, {key: float(value) for key, value in result.items()})
                checked.append(f'{kind}:n={count}')
        finally:
            trainer.close()
    return {'checked': checked, 'reference': 'one_gpu_full_dataset'}


def metric_owner_case(context, precision):
    size = context.world_size * 2 + 1
    logits, targets = classification_data(size)
    dataset = TensorDataset(logits, targets)
    # 非零rank共同覆盖完整集合，rank0完全没有数据。
    indices = [] if context.rank == 0 else list(range(size))[context.rank - 1::context.world_size - 1]
    for kind in ('tensor', 'confusion'):
        trainer = make_trainer(context, precision, metrics=[weighted_metrics(context, kind)], metric_patch=kind)
        try:
            result = quiet(trainer.test, DataLoader(dataset, batch_size=2, sampler=ExactIndicesSampler(indices)))
            expected = weighted_metrics(context, 'tensor')(logits.to(context.device), targets.to(context.device))
            assert_close_number(result['loss'], F.cross_entropy(logits.to(context.device), targets.to(context.device)))
            for name in expected:
                assert_close_number(result[name], expected[name])
        finally:
            trainer.close()

    def ratio(correct, count):
        assert correct.device == context.device and count.device == context.device
        return correct / count

    class RemoteAccuracy(PatchBase):
        def __init__(self):
            super().__init__('remote_accuracy')
            self.metric = ratio

        def load(self, preds, labels):
            self.correct = (preds.argmax(1) == labels).sum().detach()
            self.count = torch.tensor(len(labels), device=preds.device)
            return self

        def forward(self):
            return self.metric(self.correct, self.count)

        def add(self, other):
            self.correct = self.correct + other.correct
            self.count = self.count + other.count
            return self

    owner = context.world_size - 1

    class DynamicTask(EpochTask):
        def evaluate_step(self, batch_x, batch_y):
            output = self.model(batch_x)
            self.loss(output, batch_y)
            return {'remote': RemoteAccuracy().load(output, batch_y)} if self.rank == owner else {}

    trainer = make_trainer(context, precision)
    try:
        result = quiet(trainer.test, tasks=DynamicTask(DataLoader(dataset, batch_size=2)))
        owner_indices = torch.arange(owner, size, context.world_size, device=context.device)
        gpu_logits, gpu_targets = logits.to(context.device), targets.to(context.device)
        expected = (gpu_logits[owner_indices].argmax(1) == gpu_targets[owner_indices]).float().mean()
        assert_close_number(result['remote_accuracy'], expected)
        assert_close_number(result['loss'], F.cross_entropy(gpu_logits, gpu_targets))
    finally:
        trainer.close()
    return {'rank_zero_empty': True, 'dynamic_metric_owner': owner, 'cuda_closure': True}


def aggregation_errors_case(context):
    class LocalMetricError(ValueError):
        pass

    class CombinedFailure(PatchBase):
        def load(self, value):
            self.value = value
            return self

        def forward(self):
            if self.value > 1:
                raise LocalMetricError('combined CUDA metric failed')
            return self.value

        def add(self, other):
            self.value = self.value + other.value
            return self

    try:
        aggregate_patch_dicts([{'bad': CombinedFailure().load(torch.tensor(1, device=context.device))}], context)
    except RuntimeError as error:
        owner_message = str(error)
        assert 'LocalMetricError' in owner_message and 'combined CUDA metric failed' in owner_message
        for detail in ("指标 'bad'", 'Traceback (most recent call last)', 'in forward', 'multigpu_evaluation.py'):
            assert detail in owner_message, owner_message
    else:
        raise AssertionError('owner exception was not propagated')
    same_on_all_ranks(context, owner_message)
    local = [{'value': ValuePatch(1).load({'a': torch.tensor(1., device=context.device)})}]
    if context.rank == 1:
        local.append({'value': ValuePatch(1).load({'b': torch.tensor(2., device=context.device)})})
    try:
        aggregate_patch_dicts(local, context)
    except RuntimeError as error:
        local_message = str(error)
        assert 'rank 1' in local_message and 'AssertionError' in local_message
        for detail in ("指标 'value'", 'Traceback (most recent call last)', 'in add', 'patches.py'):
            assert detail in local_message, local_message
    else:
        raise AssertionError('local merge exception was not propagated')
    same_on_all_ranks(context, local_message)
    return {'owner_error': 'all_ranks_received', 'local_error': 'all_ranks_received',
            'source_locations_preserved': True}



def reused_task_lists_case(context, precision, output):
    class RecordingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, optimizer):
            self.losses = []
            super().__init__(optimizer, patience=0, factor=0.5)

        def step(self, loss, epoch=None):
            self.losses.append(float(loss))
            super().step(loss, epoch)

    class TaskOrder(Callback):
        def __init__(self):
            super().__init__()
            self.validation, self.testing = [], []

        def on_before_val_epoch(self, trainer, task):
            self.validation.append(task)

        def on_before_test_epoch(self, trainer, task):
            self.testing.append(task)

    def auxiliary(preds, targets):
        return preds.new_tensor(7.)

    # 零输入固定模型输出和训练梯度；新验证集的全局参考损失为1、9。
    inputs = torch.zeros(context.world_size * 2 + 1, 1)
    train_loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
    loaders = [DataLoader(TensorDataset(inputs, torch.full_like(inputs, target)), batch_size=2)
               for target in (1., 3.)]
    val_task = EpochTask(train_loader, metrics=[auxiliary], do_loss=False, name='aux')
    test_task = EpochTask(train_loader, metrics=[auxiliary], do_loss=False, name='aux')
    val_tasks, test_tasks = [val_task], [test_task]
    order = TaskOrder()
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    raw = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingPlateau(raw)
    checker = CheckCallback('loss', on_stage='val', patience=1, ckpt_dir=str(output / 'reused-task-lists'))
    trainer = make_trainer(context, precision, model=model, loss=F.mse_loss,
                           opt=Optimizer(raw, scheduler, sched_with_loss=True, monitor='loss', on_stage='val'),
                           callbacks=[checker, order], epochs=1, running_id='reused')

    def assert_order(observed, task):
        assert len(observed) == 4, len(observed)
        assert observed[0].dataloader is loaders[0] and observed[1] is task
        assert observed[2].dataloader is loaders[1] and observed[3] is task

    try:
        first = quiet(trainer.fit, train_loader, loaders[0], val_tasks=val_tasks)
        # 第二次fit允许额外三轮，验证损失恶化后必须只执行一轮。
        second = quiet(trainer.fit, train_loader, loaders[1], val_tasks=val_tasks, epochs=3)
        validation_losses = [float(first['val']['loss'][0]), float(second['val']['loss'][0])]
        assert validation_losses == scheduler.losses == [1., 9.]
        assert len(second['val']['loss']) == 1 and trainer.init_epoch == 2
        assert checker.best_value == 1. and checker.worse_times == 1
        assert_close_number(raw.param_groups[0]['lr'], 0.05)
        assert [float(first['val']['aux/auxiliary'][0]), float(second['val']['aux/auxiliary'][0])] == [7., 7.]
        assert len(val_tasks) == 1 and val_tasks[0] is val_task
        assert_order(order.validation, val_task)

        scaler_before_test = deepcopy(trainer.scaler.state_dict())
        results = [quiet(trainer.test, loader, tasks=test_tasks) for loader in loaders]
        testing_losses = [float(result['loss']) for result in results]
        assert testing_losses == [1., 9.]
        assert [float(result['aux/auxiliary']) for result in results] == [7., 7.]
        assert len(test_tasks) == 1 and test_tasks[0] is test_task
        assert_order(order.testing, test_task)
        assert trainer.init_epoch == 2 and scheduler.losses == [1., 9.]
        assert trainer.scaler.state_dict() == scaler_before_test
        assert_close_number(model.weight.detach(), 0.25)
        result = {'validation_losses': validation_losses, 'testing_losses': testing_losses,
                  'scheduler_losses': scheduler.losses, 'epochs': trainer.init_epoch,
                  'second_fit_epochs': len(second['val']['loss']),
                  'early_stop_worse_times': checker.worse_times,
                  'learning_rate': raw.param_groups[0]['lr'],
                  'caller_list_sizes': [len(val_tasks), len(test_tasks)],
                  'task_order': ['loader', 'task', 'loader', 'task'],
                  'weight': float(model.weight.detach()), 'scaler': trainer.scaler.state_dict()}
        same_on_all_ranks(context, result)
        return result
    finally:
        trainer.close()


def checkpoint_case(context, precision, output):
    class AdvanceMarker(Callback):
        def on_after_epoch(self, trainer, **kwargs):
            with torch.no_grad():
                context.unwrap_model(trainer.model.model).anchor.add_(1.)

    def train_f1(preds, labels):
        return metrics.f1(preds, labels, average='macro')

    def val_score(preds, labels):
        return (preds.argmax(1) == labels).float().mean()

    train_logits, train_targets = classification_data(context.world_size * 2)
    val_logits, val_targets = classification_data(context.world_size * 2 + 1)
    train_loader = DataLoader(TensorDataset(train_logits, train_targets), batch_size=2)
    val_loader = DataLoader(TensorDataset(val_logits, val_targets), batch_size=2)
    tiny_loader = DataLoader(TensorDataset(val_logits[:1], val_targets[:1]), batch_size=2)
    model = FixedLogits()
    raw = torch.optim.SGD(model.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(raw, patience=0, factor=0.5)
    first = CheckCallback('full/loss', on_stage='val', patience=2, save_best=True, ckpt_dir=str(output / 'checkpoints'))
    second = CheckCallback('full/val_score', on_stage='val', mode='max', save_best=False, ckpt_dir=str(output / 'checkpoints'))
    trainer = make_trainer(context, precision, model=model,
                           opt=Optimizer(raw, scheduler, sched_with_loss=True, monitor='full/loss', on_stage='val'), epochs=8,
                           callbacks=[first, second, AdvanceMarker()], running_id='multiple')
    save = torch.save
    saved_paths = []

    def record_save(state, path, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            saved_paths.append(str(path))
        return save(state, path, **kwargs)

    try:
        with patch('deepepochs.callbacks.check.torch.save', side_effect=record_save):
            history = quiet(trainer.fit,
                            train_tasks=[EpochTask(train_loader, metrics=[train_f1], name='primary'),
                                         EpochTask(train_loader, name='auxiliary')],
                            val_tasks=[EpochTask(tiny_loader, name='tiny'),
                                       EpochTask(val_loader, metrics=[val_score], name='full')])
        assert len(history['train']['primary/loss']) == 3 and trainer.init_epoch == 3
        assert len(saved_paths) == (4 if context.is_main_process else 0), saved_paths
        expected_loss = F.cross_entropy(val_logits.to(context.device), val_targets.to(context.device))
        assert_close_number(history['val']['full/loss'][-1], expected_loss)
        assert set(history['train']) == {'primary/loss', 'primary/train_f1', 'auxiliary/loss'}
        assert set(history['val']) == {'tiny/loss', 'full/loss', 'full/val_score'}
        assert history['train']['primary/loss'] == history['train']['auxiliary/loss']
        assert_close_number(history['val']['tiny/loss'][-1],
                            F.cross_entropy(val_logits[:1].to(context.device), val_targets[:1].to(context.device)))
        assert_close_number(scheduler.best, expected_loss)
        assert_close_number(raw.param_groups[0]['lr'], 0.02)
        assert first.worse_times == second.worse_times == 2
        assert first.ckpt_path != second.ckpt_path
        states = [torch.load(path, map_location='cpu', weights_only=False) for path in (first.ckpt_path, second.ckpt_path)]
        assert [float(state['model_state']['anchor']) for state in states] == [1., 3.]
        assert [state['epoch'] for state in states] == [0, 2]
        assert all(set(state['model_state']) == {'anchor', 'bias'} for state in states)
        assert [entry['worse_times'] for entry in states[0]['callback_states']] == [0, 0]
        assert [entry['worse_times'] for entry in states[1]['callback_states']] == [2, 2]
        original_epoch = trainer.init_epoch
        original_scaler = deepcopy(trainer.scaler.state_dict())
        original_scheduler = deepcopy(scheduler.state_dict())
        seen = []

        def observe_best(preds, labels):
            seen.append(float(model.anchor.detach()))
            return preds.mean()

        quiet(trainer.test, val_loader, metrics=[observe_best])
        assert seen and all(value == 1. for value in seen), seen
        assert float(model.anchor.detach()) == 3.
        assert trainer.init_epoch == original_epoch and trainer.scaler.state_dict() == original_scaler
        assert scheduler.state_dict() == original_scheduler
        same_on_all_ranks(context, (trainer.init_epoch, first.worse_times, second.worse_times,
                                   float(model.anchor.detach()), raw.param_groups[0]['lr']))
        return {'epochs': 3, 'learning_rate': raw.param_groups[0]['lr'],
                'best_markers': [1, 3], 'portable_checkpoint': second.ckpt_path,
                'rank_zero_writes': 4, 'test_restored_current_model': True}
    finally:
        trainer.close()


def multiple_checkpoint_directories_case(context, precision, output):
    root = output / 'multiple-checkpoint-directories'
    first = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=str(root / 'a'))
    second = CheckCallback('score', on_stage='train', mode='max', ckpt_dir=str(root / 'b'))
    model = FixedLogits()
    source = make_trainer(context, precision, model=model, callbacks=[first, second], running_id='saved')
    try:
        for epoch in (0, 1):
            with torch.no_grad():
                model.anchor.fill_(epoch + 1)
            first.do_check(source, {'loss': float(epoch + 1)}, epoch, defer=True)
            second.do_check(source, {'score': 1. / (epoch + 1)}, epoch, defer=True)
            finalize_checkpoints(source)
        # A保存最近一轮，B保留第一轮最佳；文件时间明确标识先后。
        if context.is_main_process:
            os.utime(first.ckpt_path, (1_800_000_000, 1_800_000_000))
            os.utime(second.ckpt_path, (1_700_000_000, 1_700_000_000))
        context.barrier()
        saved_markers = [float(torch.load(callback.ckpt_path, map_location='cpu', weights_only=False)
                               ['model_state']['anchor']) for callback in (first, second)]
        assert saved_markers == [2., 1.], saved_markers
    finally:
        source.close()

    callbacks = [CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=first.ckpt_dir),
                 CheckCallback('score', on_stage='train', mode='max', ckpt_dir=second.ckpt_dir)]
    # 旧快照所在的B监控器先进入before_fit，仍必须从全部目录选择A。
    callbacks[0].priority, callbacks[1].priority = -1, -2
    restored_model = FixedLogits()
    restored = make_trainer(context, precision, model=restored_model, callbacks=callbacks,
                            resume=True, epochs=2, running_id='continued')
    try:
        logits, targets = classification_data(context.world_size * 2)
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        # 快照已完成两轮，fit只执行恢复初始化，不产生新的参数更新。
        history = quiet(restored.fit, loader)
        assert history == {} and restored.init_epoch == 2
        assert float(restored_model.anchor.detach()) == 2.
        paths = [callback.ckpt_path for callback in callbacks]
        assert paths == [first.ckpt_path, second.ckpt_path], paths
        assert [callback.best_value for callback in callbacks] == [1., 1.]
        assert [callback.worse_times for callback in callbacks] == [1, 1]
        assert all(callback._resume_loaded for callback in callbacks)
        same_on_all_ranks(context, (restored.init_epoch, float(restored_model.anchor.detach()), paths,
                                   [callback.worse_times for callback in callbacks]))
        return {'restored_epoch': restored.init_epoch, 'restored_marker': 2.,
                'saved_markers': saved_markers, 'first_monitor_to_resume': 'score',
                'selected_checkpoint': first.ckpt_path, 'monitor_paths': paths}
    finally:
        restored.close()


def custom_stop_case(context, precision, output):
    class StopBetweenMonitors(Callback):
        def __init__(self):
            super().__init__()
            self.priority = 1
            self.enabled = True

        def on_after_epoch(self, **kwargs):
            if self.enabled:
                raise StopLoopException('custom distributed stop')

    first = CheckCallback('loss', on_stage='train', patience=1, save_best=False, ckpt_dir=str(output / 'custom-stop'))
    second = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=str(output / 'custom-stop'))
    first.priority, second.priority, first.best_value = 2, -1, 0.
    stopper = StopBetweenMonitors()
    logits, targets = classification_data(context.world_size * 2)
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    trainer = make_trainer(context, precision, callbacks=[first, stopper, second], epochs=1, running_id='stop')
    save = torch.save
    saved_epochs = []

    def record_save(state, path, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            saved_epochs.append(state['epoch'])
        return save(state, path, **kwargs)

    try:
        capture = io.StringIO()
        with patch('deepepochs.callbacks.check.torch.save', side_effect=record_save):
            with contextlib.redirect_stdout(capture):
                trainer.fit(loader)
            assert trainer._pending_checkpoints == [] and second.ckpt_path is None
            if context.is_main_process:
                assert 'custom distributed stop' in capture.getvalue()
                assert 'Early stopping triggered' not in capture.getvalue()
            stopper.enabled, first.patience = False, 0
            quiet(trainer.fit, loader, epochs=1)
        assert saved_epochs == ([0, 1, 1] if context.is_main_process else []), saved_epochs
        same_on_all_ranks(context, (len(trainer._pending_checkpoints), trainer.init_epoch, first.worse_times))
        return {'pending': 0, 'epochs_saved': [0, 1, 1], 'original_stop_preserved': True}
    finally:
        trainer.close()


def interpreter_case(context, precision, output):
    logits, targets = classification_data(context.world_size * 2 + 1)
    callback = InterpreteCallback(metric=lambda preds, labels: F.cross_entropy(preds, labels, reduction='none'),
                                 k=3, stages='test', log_dir=str(output / 'tensorboard'))
    trainer = make_trainer(context, precision, callbacks=[callback], running_id='interpretation')
    try:
        with patch('deepepochs.callbacks.interprete.SummaryWriter', wraps=SummaryWriter) as writer:
            quiet(trainer.test, DataLoader(TensorDataset(logits, targets), batch_size=2))
            assert writer.call_count == int(context.is_main_process)
        expected_matrix = metrics.confusion_matrix(logits.to(context.device), targets.to(context.device), 2).cpu()
        torch.testing.assert_close(callback.confusion_matrix.cpu(), expected_matrix)
        assert int(callback.confusion_matrix.sum()) == len(targets)
        expected_losses = F.cross_entropy(logits.to(context.device), targets.to(context.device), reduction='none')
        expected_indices = expected_losses.topk(3).indices.cpu()
        if context.is_main_process:
            samples = callback.top_samples()
            assert len(samples) == 3
            for sample, index in zip(samples, expected_indices):
                assert_close_number(sample[0], expected_losses[index])
                torch.testing.assert_close(sample[1], logits[index])
                torch.testing.assert_close(sample[2], targets[index])
                torch.testing.assert_close(sample[3], logits[index])
        else:
            assert not hasattr(callback, 'logger')
        same_on_all_ranks(context, int(callback.confusion_matrix.sum()))
        return {'samples': len(targets), 'top_indices': expected_indices.tolist(),
                'global_confusion_matrix': expected_matrix.tolist(), 'writer_rank': 0}
    finally:
        trainer.close()


def log_graph_analysis_case(context, precision, output, mode):
    class BranchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('offset', torch.tensor([0.25, -0.5, 0.125]))
            self.train_branch = nn.Sequential(nn.Linear(3, 4), nn.ReLU(inplace=True), nn.Linear(4, 2))
            self.eval_branch = nn.Sequential(nn.Linear(3, 4), nn.ReLU(inplace=True), nn.Linear(4, 2))
            with torch.no_grad():
                for index, parameter in enumerate(self.parameters()):
                    values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
                    parameter.copy_((values.remainder(7) - 3) / 16 + index / 128)

        def forward(self, inputs):
            branch = self.train_branch if self.training else self.eval_branch
            return branch(inputs + self.offset)

    metric_dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]

    def prediction_mean(preds, targets):
        assert preds.dtype == metric_dtype
        return preds.mean()

    # 每个rank取得完全相同的样本，参考与回调运行使用相同的精度和归约。
    inputs = torch.tensor([[1., -0.5, 0.25]]).repeat(context.world_size * 2, 1)
    targets = torch.tensor([[0.25, -0.25]]).repeat(context.world_size * 2, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    initial_model = BranchModel()
    initial_state = deepcopy(initial_model.state_dict())
    reference = BranchModel()
    reference.load_state_dict(initial_state)
    reference_opt = torch.optim.SGD(reference.parameters(), lr=0.05)
    with make_trainer(context, precision, model=reference, loss=F.mse_loss, opt=reference_opt,
                      metrics=[prediction_mean], epochs=1,
                      ddp_kwargs={'find_unused_parameters': True}) as trainer:
        quiet(trainer.fit, loader)
        expected_state = deepcopy(trainer.model.state_dict())
        expected_scaler = deepcopy(trainer.scaler.state_dict())
        expected_test_metrics = quiet(trainer.test, loader)
        for name, expected in expected_state.items():
            torch.testing.assert_close(trainer.model.state_dict()[name], expected, rtol=0, atol=0)
        assert trainer.scaler.state_dict() == expected_scaler and trainer.init_epoch == 1

    model = BranchModel()
    model.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    log_dir = output / 'tensorboard'
    logger = LogCallback(log_dir=str(log_dir), log_graph=True)
    analysis = AnalyzeCallback(mode=mode, log_dir=str(log_dir))
    with contextlib.ExitStack() as stack:
        with make_trainer(context, precision, model=model, loss=F.mse_loss, opt=optimizer,
                          metrics=[prediction_mean], callbacks=[logger, analysis], epochs=1,
                          hyper_params={'precision': precision, 'analysis_mode': mode},
                          running_id=f'inspection-{mode}-{time.time_ns()}',
                          ddp_kwargs={'find_unused_parameters': True}) as trainer:
            assert not trainer.distributed._owns_process_group
            quiet(trainer.fit, loader)
            actual_state = trainer.model.state_dict()
            for name, expected in expected_state.items():
                torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
            assert trainer.scaler.state_dict() == expected_scaler
            assert all(module.training for module in model.modules())
            assert any(not torch.equal(actual_state[name].cpu(), initial_state[name])
                       for name in initial_state if name.startswith('train_branch.'))
            state = {name: value.cpu().tolist() for name, value in actual_state.items()}
            same_on_all_ranks(context, {'state': state, 'scaler': trainer.scaler.state_dict()})
            test_metrics = quiet(trainer.test, loader)
            for name, expected in expected_state.items():
                torch.testing.assert_close(trainer.model.state_dict()[name], expected, rtol=0, atol=0)
            assert trainer.scaler.state_dict() == expected_scaler and trainer.init_epoch == 1
            assert test_metrics.keys() == expected_test_metrics.keys()
            for name, expected in expected_test_metrics.items():
                assert_close_number(test_metrics[name], expected, tolerance=0)
            same_on_all_ranks(context, {name: float(value) for name, value in test_metrics.items()})
            if context.is_main_process:
                assert logger.graph_saved and logger.logger is analysis.logger is trainer.logger
                writer = trainer.logger
                assert writer.file_writer is not None
                close_writer = stack.enter_context(patch.object(writer, 'close', wraps=writer.close))
                event_dir = log_dir / trainer.running_id
                active_indices = {str(index) for index, module in enumerate(trainer.model.modules())
                                  if module in model.train_branch.modules()}
                expected_histograms = {f'{mode.upper()}-hist-{hook.name}' for hook in analysis.hooks
                                       if hook.name.split('-', 1)[0] in active_indices}
                assert len(expected_histograms) == (3 if mode == 'forward' else 2)
                first_linear_index = next(str(index) for index, module in enumerate(trainer.model.modules())
                                          if module is model.train_branch[0])
            else:
                assert not hasattr(trainer, 'logger')
                assert not hasattr(logger, 'logger') and not hasattr(analysis, 'logger')
        if context.is_main_process:
            assert close_writer.call_count == 1 and writer.file_writer is None

    assert dist.is_initialized() == context.is_distributed
    context.barrier()
    if context.is_main_process:
        assert list(event_dir.glob('events.out.tfevents.*'))
        events = EventAccumulator(str(event_dir), size_guidance={'histograms': 0}).Reload()
        assert events.Tags()['graph'] and len(events.Graph().node) > 0
        assert {'prediction_mean/batch/train', 'prediction_mean/epoch/train', 'prediction_mean'} <= set(events.Tags()['scalars'])
        assert '_hparams_/session_start_info' in events.PluginTagToContent('hparams')
        assert_close_number(events.Scalars('prediction_mean')[0].value,
                            expected_test_metrics['prediction_mean'], tolerance=0)
        assert set(events.Tags()['histograms']) == expected_histograms
        for name in expected_histograms:
            assert [event.step for event in events.Histograms(name)] == [1]
        if mode == 'forward':
            first_linear = next(name for name in expected_histograms
                                if name.startswith(f'FORWARD-hist-{first_linear_index}-'))
            assert events.Histograms(first_linear)[0].histogram_value.min < 0
    return {'mode': mode, 'training_histograms': 3 if mode == 'forward' else 2,
            'graph_saved': True, 'reference_weights_equal': True,
            'scaler': expected_scaler, 'writer_closed': True, 'borrowed_context_preserved': True,
            'metric_dtype': str(metric_dtype), 'hparams_logged': True, 'test_state_preserved': True}


def stopped_fit_resume_case(context, precision):
    class StopFirstEpoch(Callback):
        def __init__(self):
            super().__init__()
            self.epochs = []

        def on_before_epoch(self, trainer, train_tasks, val_tasks, epoch_idx):
            self.epochs.append(epoch_idx)

        def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
            if epoch_idx == 0:
                raise StopLoopException('stop after the first completed epoch')

    logits, targets = classification_data(context.world_size * 2)
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    stopper = StopFirstEpoch()
    with make_trainer(context, precision, callbacks=[stopper], epochs=3) as trainer:
        first = quiet(trainer.fit, loader)
        assert len(first['train']['loss']) == 1 and trainer.init_epoch == 1 and trainer.max_epochs == 3
        same_on_all_ranks(context, trainer.init_epoch)
        second = quiet(trainer.fit, loader, epochs=1)
        assert len(second['train']['loss']) == 1 and trainer.init_epoch == trainer.max_epochs == 2
        assert stopper.epochs == [0, 1]
        same_on_all_ranks(context, (trainer.init_epoch, trainer.max_epochs, stopper.epochs))
    return {'initial_target_epochs': 3, 'first_fit_epochs': 1, 'second_fit_epochs': 1,
            'completed_epochs': 2, 'epoch_indices': [0, 1]}


def named_dataset_monitors_case(context, precision, output):
    class RecordingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, optimizer):
            self.losses = []
            super().__init__(optimizer, patience=0, factor=0.5)

        def step(self, loss, epoch=None):
            self.losses.append(float(loss))
            super().step(loss, epoch)

    inputs = torch.zeros(context.world_size * 2 + 1, 1)

    def tasks(targets):
        return [EpochTask(DataLoader(TensorDataset(inputs, torch.full_like(inputs, target)), batch_size=2),
                          name=name) for name, target in zip(('source', 'domain'), targets)]

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    raw = [torch.optim.SGD(model.parameters(), lr=0.1) for _ in range(2)]
    schedulers = [RecordingPlateau(optimizer) for optimizer in raw]
    optimizers = [Optimizer(raw[0], schedulers[0], sched_with_loss=True,
                            monitor='source/loss', on_stage='train'),
                  Optimizer(raw[1], schedulers[1], sched_with_loss=True,
                            monitor='domain/loss', on_stage='val')]
    checker = CheckCallback('domain/loss', on_stage='val', save_best=False,
                            ckpt_dir=str(output / 'named-dataset-monitors'))
    trainer = make_trainer(context, precision, model=model, loss=F.mse_loss, opt=optimizers,
                           callbacks=[checker], epochs=3, running_id='named')
    try:
        history = quiet(trainer.fit, train_tasks=tasks((1., 3.)), val_tasks=tasks((2., 4.)), val_freq=2)
        assert set(history['train']) == set(history['val']) == {'source/loss', 'domain/loss'}
        assert list(history['train']['source/loss']) == [1.] * 3
        assert list(history['train']['domain/loss']) == [9.] * 3
        assert list(history['val']['source/loss']) == [4.]
        assert list(history['val']['domain/loss']) == [16.]
        assert [scheduler.losses for scheduler in schedulers] == [[1., 1., 1.], [16.]]
        assert checker.best_value == 16.
        actual = quiet(trainer.test, tasks=tasks((2., 4.)), weights='current')
        assert {key: float(value) for key, value in actual.items()} == {'source/loss': 4., 'domain/loss': 16.}
        result = {'train_losses': [1., 9.], 'validation_losses': [4., 16.],
                  'epoch_monitors': ['train/source/loss', 'val/domain/loss'],
                  'scheduler_losses': [scheduler.losses for scheduler in schedulers],
                  'validation_epochs': 1, 'checkpoint_monitor': checker.monitor}
        same_on_all_ranks(context, result)
        return result
    finally:
        trainer.close()


def unlabeled_prediction_case(context, precision, output):
    class PredictionTrainer(Trainer):
        def predict_step(self, batch_x):
            return {'sample_id': batch_x['sample_id'], 'value': self.model(batch_x['x'])}

    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Dropout(0.2))
    model[0].weight.data.fill_(2.)
    raw = torch.optim.SGD(model.parameters(), lr=0.)
    checker = CheckCallback('loss', on_stage='train', save_best=True,
                            ckpt_dir=str(output / 'prediction-weights'))
    trainer = PredictionTrainer(model, F.mse_loss, raw, callbacks=[checker], device=context.device,
                                precision=precision, non_blocking=True, show_info=False, log_batch=False, running_id='selected')
    try:
        train_x = torch.ones(context.world_size * 2, 1)
        quiet(trainer.fit, DataLoader(TensorDataset(train_x, 2 * train_x), batch_size=2), epochs=1)
        model[0].weight.data.fill_(3.)
        model.train()
        model[1].eval()
        trainer.model.stage = 'train'
        previous_task = trainer.current_task
        previous_modes = [module.training for module in trainer.model.modules()]
        previous_epoch = trainer.init_epoch
        previous_scaler = deepcopy(trainer.scaler.state_dict())
        selections = [('current', 'current', 3.), ('auto', 'auto', 2.),
                      ('callback', checker, 2.), ('path', checker.ckpt_path, 2.)]
        checked = []
        for size in (context.world_size * 2 + 1, 1):
            ids = torch.arange(size)
            inputs = ids.float().view(-1, 1) / 8
            loader = DataLoader([{'sample_id': index, 'x': value} for index, value in zip(ids, inputs)],
                                batch_size=2, pin_memory=context.device.type == 'cuda')
            for label, weights, factor in selections:
                batches = quiet(lambda: list(trainer.predict(loader, weights=weights)))
                local_ids = torch.cat([batch['sample_id'] for batch in batches]) if batches else torch.empty(0, dtype=torch.int64)
                local_values = torch.cat([batch['value'] for batch in batches]) if batches else torch.empty(0, 1)
                assert local_ids.tolist() == list(range(context.rank, size, context.world_size))
                assert all(batch['value'].device.type == 'cpu' and not batch['value'].requires_grad for batch in batches)
                gathered = context.all_gather_object((local_ids, local_values))
                merged_ids = torch.cat([item[0] for item in gathered])
                merged_values = torch.cat([item[1] for item in gathered])
                order = merged_ids.argsort()
                assert merged_ids[order].tolist() == ids.tolist()
                with trainer.autocast(), torch.no_grad():
                    reference = F.linear(inputs.to(context.device), torch.full((1, 1), factor, device=context.device))
                torch.testing.assert_close(merged_values[order].float(), reference.cpu().float(), atol=0, rtol=0)
                assert [module.training for module in trainer.model.modules()] == previous_modes
                assert trainer.model.stage == 'train' and trainer.current_task is previous_task
                assert float(model[0].weight.detach()) == 3.
                checked.append(f'{label}:n={size}')

        # 各进程都取一批后主动关闭；临时权重、混合模式与训练上下文立即恢复。
        loader = DataLoader([{'sample_id': index, 'x': torch.tensor([index / 8.])}
                             for index in range(context.world_size * 3)], batch_size=1)
        stream = trainer.predict(loader, weights=checker)
        first = quiet(next, stream)
        assert first['sample_id'].tolist() == [context.rank]
        assert float(model[0].weight.detach()) == 2.
        stream.close()
        assert float(model[0].weight.detach()) == 3.
        assert [module.training for module in trainer.model.modules()] == previous_modes
        assert trainer.model.stage == 'train' and trainer.current_task is previous_task
        assert trainer.init_epoch == previous_epoch and trainer.scaler.state_dict() == previous_scaler
        return {'checked': checked, 'unpadded_ids': True, 'full_dataset_reference': True,
                'early_close_restores_current_weights': True, 'mixed_modes_restored': True}
    finally:
        trainer.close()


def _assert_cpu_tensors(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu' and not value.requires_grad
    elif isinstance(value, dict):
        for item in value.values():
            _assert_cpu_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_cpu_tensors(item)


def _assert_same_state(actual, expected, floating_tolerance=False):
    if isinstance(actual, torch.Tensor):
        tolerance = {"rtol": 1e-6, "atol": 1e-7} if floating_tolerance and actual.is_floating_point() else {"rtol": 0, "atol": 0}
        torch.testing.assert_close(actual.cpu(), expected.cpu(), **tolerance)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_same_state(actual[key], expected[key], floating_tolerance)
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_same_state(value, reference, floating_tolerance)
    else:
        assert actual == expected, (actual, expected)


def stochastic_epoch_resume_case(context, precision, output):
    class RandomDataset(Dataset):
        def __len__(self):
            return context.world_size * 6

        def __getitem__(self, index):
            noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
            return torch.tensor([(index / len(self) + noise) / 4]), torch.tensor([index / len(self)])

    class DeviceCounter(Callback):
        def __init__(self):
            super().__init__()
            self.count = torch.zeros((), device=context.device)

        def on_before_fit(self, trainer, epochs):
            self.count.zero_()
            random.random()
            np.random.rand()
            torch.rand(())
            if context.device.type == 'cuda':
                torch.rand((), device=context.device)
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.fill_(0.25)

        def on_after_train_batch(self, trainer, metrics, batch_idx):
            self.count.add_(1)

        def state_dict(self):
            return {'count': self.count}

        def load_state_dict(self, state):
            self.count = state['count'].to(context.device)

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if context.device.type == 'cuda':
            torch.cuda.manual_seed(value)

    def loader(value):
        return DataLoader(RandomDataset(), batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(value + context.rank))

    trainers = []

    def training(root, running_id, resume=False):
        model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Dropout(0.3), nn.Linear(4, 1))
        raw = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(raw, step_size=1, gamma=0.8)
        checker = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=str(root))
        counter = DeviceCounter()
        trainer = make_trainer(context, precision, model=model, loss=F.mse_loss,
                               opt=Optimizer(raw, scheduler), callbacks=[checker, counter],
                               running_id=running_id, resume=resume)
        trainers.append(trainer)
        return trainer, checker, counter

    def random_sample():
        return (random.random(), float(np.random.rand()), torch.rand(()), torch.rand((), device=context.device))

    root = output / 'stochastic-epoch-resume'
    try:
        seed(20 + context.rank)
        full, _, full_counter = training(root / 'full', 'full')
        full_history = quiet(full.fit, loader(80), epochs=3)
        expected_model = deepcopy(full.model.state_dict())
        expected_opt = deepcopy(full.opt.state_dict())
        expected_scaler = deepcopy(full.scaler.state_dict())
        expected_count = float(full_counter.count)
        expected_random = random_sample()
        expected_generator = full.train_tasks[0]._train_dataloader.generator.get_state().clone()

        seed(20 + context.rank)
        source, checker, source_counter = training(root / 'split', 'saved')
        save = torch.save
        observed_rank_states = []

        def record_save(state, path, **kwargs):
            if isinstance(path, (str, os.PathLike)) and 'rank_states' in state:
                _assert_cpu_tensors(state['rank_states'])
                observed_rank_states.append(len(state['rank_states']))
            return save(state, path, **kwargs)

        with patch('deepepochs.callbacks.check.torch.save', side_effect=record_save):
            quiet(source.fit, loader(80), epochs=1)
        assert observed_rank_states == ([context.world_size] if context.is_main_process else [])
        assert source_counter.count.device == context.device
        state = torch.load(checker.ckpt_path, map_location='cpu', weights_only=False)
        assert state['world_size'] == len(state['rank_states']) == context.world_size
        _assert_cpu_tensors(state['rank_states'])
        assert not torch.equal(state['rank_states'][0]['random']['torch'],
                               state['rank_states'][1]['random']['torch'])

        entries = ('automatic', 'explicit')
        for entry in entries:
            seed(900 + context.rank)
            resumed, _, resumed_counter = training(root / 'split', entry,
                                                   resume='saved' if entry == 'automatic' else False)
            if entry == 'explicit':
                quiet(resumed.load_checkpoint, checker.ckpt_path)
            resumed_history = quiet(resumed.fit, loader(700), epochs=3)
            _assert_same_state(resumed.model.state_dict(), expected_model, floating_tolerance=True)
            _assert_same_state(resumed.opt.state_dict(), expected_opt, floating_tolerance=True)
            _assert_same_state(resumed.scaler.state_dict(), expected_scaler)
            assert float(resumed_counter.count) == expected_count
            assert resumed_counter.count.device == context.device
            np.testing.assert_allclose(resumed_history['train']['loss'], full_history['train']['loss'][1:],
                                       rtol=1e-6, atol=1e-7)
            actual_random = random_sample()
            assert actual_random[:2] == expected_random[:2]
            _assert_same_state(actual_random[2:], expected_random[2:])
            _assert_same_state(resumed.train_tasks[0]._train_dataloader.generator.get_state(), expected_generator)
        result = {'reference': f'uninterrupted_ddp_{precision}', 'epochs': 3, 'split_after_epoch': 1,
                  'world_size_preserved': context.world_size, 'weights_match': True,
                  'optimizer_state_match': True, 'scaler_exact': True, 'python_numpy_torch_rng_exact': True,
                  'loader_generator_exact': True, 'callback_count': expected_count,
                  'callback_tensor_device': str(context.device), 'saved_rank_states_cpu': True,
                  'resume_entries': list(entries)}
        return result
    finally:
        for trainer in trainers:
            trainer.close()


def epoch_metrics_and_lr_finder_case(context, precision):
    def auroc(predictions, labels):
        positives = labels.bool()
        if not positives.any() or positives.all():
            raise ValueError('AUROC requires both classes')
        scores = predictions[:, 1] - predictions[:, 0]
        positive_scores, negative_scores = scores[positives], scores[~positives]
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        return (comparisons.gt(0).float() + comparisons.eq(0).float() * 0.5).mean()

    class BatchKeys(Callback):
        def __init__(self):
            super().__init__()
            self.keys = []

        def on_after_test_batch(self, trainer, metrics, batch_idx):
            self.keys.append(set(metrics))

        def on_after_train_batch(self, trainer, metrics, batch_idx):
            self.keys.append(set(metrics))

    # 每批与每个非空评估分片都只有单类，AUROC只能在全局数据上计算。
    logits = torch.tensor([[0.15, 0.], [0.1, 0.], [0., 0.1], [0., 0.15]])
    labels = torch.tensor([0, 0, 1, 1])
    indices = [0, 1] if context.rank == 0 else [2, 3] if context.rank == 1 else []
    evaluation_loader = DataLoader(TensorDataset(logits, labels), batch_size=1,
                                   sampler=ExactIndicesSampler(indices))
    recorder = BatchKeys()
    trainer = make_trainer(context, precision, metrics=[auroc], callbacks=[recorder],
                           compute_batch_metrics=False)
    try:
        result = quiet(trainer.test, tasks=[EpochTask(evaluation_loader, name='validation')])
        assert float(result['validation/auroc']) == 1.
        assert_close_number(result['validation/loss'], F.cross_entropy(logits, labels))
        assert recorder.keys == [{'validation/loss'}] * len(indices), recorder.keys
        shard_sizes = context.all_gather_object(len(indices))
        assert shard_sizes == [2, 2] + [0] * (context.world_size - 2), shard_sizes
        same_on_all_ranks(context, {key: float(value) for key, value in result.items()})
    finally:
        trainer.close()

    # 各训练分片等长，单样本batch仍不可计算AUROC；首轮结束后才有全局指标。
    train_labels = torch.arange(context.world_size * 2).remainder(2)
    train_logits = F.one_hot(train_labels, num_classes=2).float() * 0.1
    train_loader = DataLoader(TensorDataset(train_logits, train_labels), batch_size=1)
    finder = LRFindCallback(max_batch=3, min_lr=1e-4, max_lr=1e-3)
    recorder = BatchKeys()
    trainer = make_trainer(context, precision, metrics=[auroc], callbacks=[finder, recorder],
                           compute_batch_metrics=False)
    try:
        with patch('deepepochs.callbacks.lr_find.plt.show') as show:
            history = quiet(trainer.fit, train_tasks=[EpochTask(train_loader, name='main')], epochs=3)
            assert show.call_count == int(context.is_main_process)
        assert recorder.keys == [{'main/loss'}] * 3, recorder.keys
        assert finder.batches_seen == 3
        assert len(history['train']['main/auroc']) == 1
        assert trainer.init_epoch == 1
        assert float(history['train']['main/auroc'][0]) == 1.
        finder_counts = context.all_gather_object({'rank': context.rank, 'batches': finder.batches_seen,
                                                   'loss_records': len(finder.losses)})
        assert all(item['batches'] == 3 for item in finder_counts), finder_counts
        assert [item['loss_records'] for item in finder_counts] == [3] + [0] * (context.world_size - 1)
        same_on_all_ranks(context, {'epochs': trainer.init_epoch, 'auroc': float(history['train']['main/auroc'][0])})
        return {'global_auroc': 1., 'single_class_evaluation_shards': True,
                'evaluation_samples_by_rank': shard_sizes, 'batch_metrics': 'loss_only',
                'finder_task': 'main', 'finder_counts': finder_counts,
                'finder_completed_epochs': trainer.init_epoch, 'finder_epoch_auroc': 1.}
    finally:
        trainer.close()
        from matplotlib import pyplot as plt
        plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--precision', default='fp32', choices=('fp32', 'fp16', 'bf16'))
    parser.add_argument('--expected-world-size', required=True, type=int, choices=(2, 3, 4))
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--cases', nargs='+', metavar='CASE')
    args = parser.parse_args()
    context = DistributedContext(args.device)
    try:
        assert context.world_size == args.expected_world_size
        assert dist.get_backend() == ('nccl' if args.device == 'cuda' else 'gloo')
        if args.precision == 'bf16' and args.device == 'cuda':
            assert torch.cuda.is_bf16_supported(), 'BF16 is unavailable on this CUDA device'
        devices = context.all_gather_object({'rank': context.rank, 'local_rank': context.local_rank,
                                             'device': str(context.device),
                                             'name': torch.cuda.get_device_name(context.device) if args.device == 'cuda' else 'cpu',
                                             'capability': torch.cuda.get_device_capability(context.device) if args.device == 'cuda' else None})
        output = args.output.resolve()
        if context.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
        context.barrier()
        metadata = {'version': __version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'nccl': torch.cuda.nccl.version() if args.device == 'cuda' else None,
                    'world_size': context.world_size,
                    'precision': args.precision, 'devices': devices}
        if context.is_main_process:
            print(json.dumps({'event': 'environment', **metadata}), flush=True)
        cases = [
            ('global_metrics', lambda: global_metrics_case(context, args.precision)),
            ('reused_task_lists', lambda: reused_task_lists_case(context, args.precision, output)),
            ('metric_owners', lambda: metric_owner_case(context, args.precision)),
            ('aggregation_errors', lambda: aggregation_errors_case(context)),
            ('multitask_scheduler_checkpoint', lambda: checkpoint_case(context, args.precision, output)),
            ('multiple_checkpoint_directories',
             lambda: multiple_checkpoint_directories_case(context, args.precision, output)),
            ('custom_stop_cleanup', lambda: custom_stop_case(context, args.precision, output)),
            ('interpreter', lambda: interpreter_case(context, args.precision, output)),
            ('log_graph_analysis_forward',
             lambda: log_graph_analysis_case(context, args.precision, output, 'forward')),
            ('log_graph_analysis_backward',
             lambda: log_graph_analysis_case(context, args.precision, output, 'backward')),
            ('stopped_fit_resume', lambda: stopped_fit_resume_case(context, args.precision)),
            ('named_dataset_monitors', lambda: named_dataset_monitors_case(context, args.precision, output)),
            ('unlabeled_prediction', lambda: unlabeled_prediction_case(context, args.precision, output)),
            ('stochastic_epoch_resume', lambda: stochastic_epoch_resume_case(context, args.precision, output)),
            ('epoch_metrics_and_lr_finder', lambda: epoch_metrics_and_lr_finder_case(context, args.precision)),
        ]
        if args.cases:
            unknown = set(args.cases) - {name for name, _ in cases}
            if unknown:
                parser.error(f'unknown cases: {", ".join(sorted(unknown))}')
            cases = [(name, case) for name, case in cases if name in args.cases]
        results = []
        for name, case in cases:
            started = time.monotonic()
            result = case()
            context.barrier()
            record = {'case': name, 'status': 'PASS', 'seconds': round(time.monotonic() - started, 3), **result}
            results.append(record)
            if context.is_main_process:
                print(json.dumps(record), flush=True)
        if context.is_main_process:
            report = {'status': 'PASS', **metadata, 'cases': results}
            device_label = 'gpu' if args.device == 'cuda' else 'cpu'
            report_path = output / f'evaluation-{context.world_size}{device_label}-{args.precision}.json'
            report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'status': 'PASS', 'report': str(report_path), 'cases': len(results)}), flush=True)
        context.barrier()
    finally:
        context.close()


if __name__ == '__main__':
    main()
