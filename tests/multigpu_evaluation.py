"""torchrun下的CUDA/NCCL评估、回调与Checkpoint联调。"""
import argparse
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from deepepochs import Callback, CheckCallback, EpochTask, InterpreteCallback, Optimizer, PatchBase, Trainer, __version__
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
    else:
        raise AssertionError('local merge exception was not propagated')
    same_on_all_ranks(context, local_message)
    return {'owner_error': 'all_ranks_received', 'local_error': 'all_ranks_received'}



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
    val_task = EpochTask(train_loader, metrics=[auxiliary], do_loss=False)
    test_task = EpochTask(train_loader, metrics=[auxiliary], do_loss=False)
    val_tasks, test_tasks = [val_task], [test_task]
    order = TaskOrder()
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    raw = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingPlateau(raw)
    checker = CheckCallback('loss', patience=1, ckpt_dir=str(output / 'reused-task-lists'))
    trainer = make_trainer(context, precision, model=model, loss=F.mse_loss,
                           opt=Optimizer(raw, scheduler, sched_with_loss=True),
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
        assert [float(first['val']['auxiliary'][0]), float(second['val']['auxiliary'][0])] == [7., 7.]
        assert len(val_tasks) == 1 and val_tasks[0] is val_task
        assert_order(order.validation, val_task)

        scaler_before_test = deepcopy(trainer.scaler.state_dict())
        results = [quiet(trainer.test, loader, tasks=test_tasks) for loader in loaders]
        testing_losses = [float(result['loss']) for result in results]
        assert testing_losses == [1., 9.]
        assert [float(result['auxiliary']) for result in results] == [7., 7.]
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
    first = CheckCallback('loss', on_stage='val', patience=2, save_best=True, ckpt_dir=str(output / 'checkpoints'))
    second = CheckCallback('val_score', on_stage='val', mode='max', save_best=False, ckpt_dir=str(output / 'checkpoints'))
    trainer = make_trainer(context, precision, model=model,
                           opt=Optimizer(raw, scheduler, sched_with_loss=True), epochs=8,
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
                            train_tasks=[EpochTask(train_loader, metrics=[train_f1]), EpochTask(train_loader)],
                            val_tasks=[EpochTask(tiny_loader), EpochTask(val_loader, metrics=[val_score])])
        assert len(history['train']['loss']) == 3 and trainer.init_epoch == 3
        assert len(saved_paths) == (4 if context.is_main_process else 0), saved_paths
        expected_loss = F.cross_entropy(val_logits.to(context.device), val_targets.to(context.device))
        assert_close_number(history['val']['loss'][-1], expected_loss)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--precision', default='fp32', choices=('fp32', 'fp16', 'bf16'))
    parser.add_argument('--expected-world-size', required=True, type=int, choices=(2, 3, 4))
    args = parser.parse_args()
    context = DistributedContext('cuda')
    try:
        assert context.world_size == args.expected_world_size
        assert dist.get_backend() == 'nccl'
        if args.precision == 'bf16':
            assert torch.cuda.is_bf16_supported(), 'BF16 is unavailable on this CUDA device'
        devices = context.all_gather_object({'rank': context.rank, 'local_rank': context.local_rank,
                                             'device': str(context.device),
                                             'name': torch.cuda.get_device_name(context.device),
                                             'capability': torch.cuda.get_device_capability(context.device)})
        output = args.output.resolve()
        if context.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
        context.barrier()
        metadata = {'version': __version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'nccl': torch.cuda.nccl.version(), 'world_size': context.world_size,
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
        ]
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
            report_path = output / f'evaluation-{context.world_size}gpu-{args.precision}.json'
            report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'status': 'PASS', 'report': str(report_path), 'cases': len(results)}), flush=True)
        context.barrier()
    finally:
        context.close()


if __name__ == '__main__':
    main()
