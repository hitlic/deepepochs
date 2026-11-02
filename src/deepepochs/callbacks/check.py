import os
import random
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from ..distributed import _DistributedEvalSampler
from ..loops import StopLoopException, _color, check_path, info
from .callback import Callback, CallbackException


def _check_callbacks(trainer):
    callbacks = [callback for callback in trainer.callbacks if isinstance(callback, CheckCallback)]
    return sorted(callbacks, key=lambda callback: getattr(callback, '_checkpoint_ordinal', 0))


def _checkpoint_path(trainer, callback):
    callbacks = _check_callbacks(trainer)
    name = 'checkpoint.ckpt' if len(callbacks) == 1 else f'checkpoint-{callbacks.index(callback) + 1}.ckpt'
    return osp.join(callback.ckpt_dir, trainer.running_id, name)


class CheckCallback(Callback):
    def __init__(self, monitor, on_stage='val', mode='min', patience=0, save_best=True, ckpt_dir='./logs', min_delta=0):
        """保存监控指标对应的Checkpoint，并按连续未改善的轮数早停。

        多个CheckCallback分别保存checkpoint-1.ckpt、checkpoint-2.ckpt；
        只有一个时使用checkpoint.ckpt。测试临时使用最佳模型，结束后恢复当前模型。

        Args:
            monitor: 指标名；on_stage指定train或val。
            mode: min或max，决定改善方向。
            patience: 连续未改善多少轮后停止，0表示不启用早停。
            save_best: True保存最佳，False保存最近一轮。
            ckpt_dir: 各次训练Checkpoint的父目录。
            min_delta: 早停基准需要超过的非负绝对改善量；最佳权重仍按严格改善保存。
        """
        assert on_stage in ['train', 'val'], 'CheckCallback的`on_stage`参数取值为"train"或"val"'
        assert mode in ['min', 'max']
        if not min_delta >= 0:
            raise ValueError('min_delta 必须是非负数。')
        self.monitor = monitor
        self.on_stage = on_stage
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.save_best = save_best
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = None
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.early_stop_baseline = self.best_value
        self.worse_times = 0
        self._checkpoint_message = None
        super().__init__(priority=-1)

    def state_dict(self):
        return {
            'best_value': self.best_value, 'early_stop_baseline': self.early_stop_baseline,
            'worse_times': self.worse_times,
        }

    def load_state_dict(self, state):
        self.best_value = state['best_value']
        self.early_stop_baseline = state.get('early_stop_baseline', self.best_value)
        self.worse_times = state['worse_times']

    def check(self, trainer, metrics, model, opt, epoch_idx, defer=False):
        """更新监控状态；defer用于等待本轮其他回调更新后统一保存。"""
        value = float(metrics[self.monitor])
        self._last_value = value
        better = value > self.best_value if self.mode == 'max' else value < self.best_value
        if better:
            self.best_value = value
        sufficient = value > self.early_stop_baseline + self.min_delta if self.mode == 'max' else \
            value < self.early_stop_baseline - self.min_delta
        if sufficient:
            self.early_stop_baseline = value
            self.worse_times = 0
        else:
            self.worse_times += 1
        self._should_save = better or not self.save_best
        self._save_path = _checkpoint_path(trainer, self) if self._should_save else None
        if self._should_save:
            if not defer:
                ordinal = _check_callbacks(trainer).index(self)
                save_state(trainer, model, opt, self._save_path,
                           checkpoint_paths={ordinal: self._save_path},
                           best_value=self.best_value, worse_times=self.worse_times, epoch=epoch_idx)
                self.ckpt_path = self._save_path
                self._record_saved(trainer, epoch_idx)
        return not (self.patience > 0 and self.worse_times >= self.patience)

    def _record_saved(self, trainer, epoch_idx):
        """成功提交文件后固定保存时的值与策略，训练结束时只汇总最后一份快照。"""
        label = 'BEST' if self.save_best else 'LATEST'
        metric = info({self.monitor: self._last_value}, trainer.default_cbk.round_to).strip()
        self._checkpoint_message = (f"{_color('[' + label + ']', 'success')} {self.on_stage} {metric}"
                                    f' | epoch={epoch_idx + 1} | checkpoint={self.ckpt_path}')

    def on_after_fit(self, trainer):
        if self._checkpoint_message is not None:
            trainer.print(self._checkpoint_message)
        elif self.ckpt_path is not None:
            trainer.print(f"{_color('[CHECKPOINT]', 'heading')} monitor: {self.on_stage}/{self.monitor}"
                          f' | checkpoint: {self.ckpt_path}')

    def on_before_fit(self, trainer, epochs):
        self._checkpoint_message = None
        if trainer.resume is False or getattr(self, '_resume_loaded', False):
            return
        callbacks = _check_callbacks(trainer)
        path = None
        if trainer.main_process:
            # 一个快照包含所有监控器状态；先汇总各保存目录的候选，
            # 再统一恢复最近的文件，不受各监控器的执行优先级影响。
            paths = []
            for directory in dict.fromkeys(callback.ckpt_dir for callback in callbacks):
                running_id = get_latest_running(directory) if trainer.resume is True else str(trainer.resume)
                if running_id is not None:
                    paths.extend(_checkpoint_paths(osp.join(directory, running_id)))
            if paths:
                path = max(paths, key=osp.getmtime)
            elif trainer.resume is not True:
                path = osp.join(callbacks[0].ckpt_dir, str(trainer.resume), 'checkpoint.ckpt')
        path = trainer.distributed.broadcast_object(path)
        if path is None:
            # resume=True也用于首次启动，目录中没有checkpoint时保留从头训练的用法。
            trainer.print(f"{_color('[RESUME]', 'heading')} No checkpoint found; starting a new training run.")
            for callback in callbacks:
                callback._resume_loaded = True
            return
        state = _broadcast_state(trainer, path)
        trainer.print(f"{_color('[RESUME]', 'heading')} checkpoint={path}")
        trainer._pending_fit_state = (state, osp.abspath(path), True)
        for callback in callbacks:
            callback._resume_loaded = True

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        if self.on_stage == 'val':
            if val_tasks and (epoch_idx + 1) % val_tasks[0].val_freq == 0:
                self.do_check(trainer, val_metrics, epoch_idx, defer=True)
        else:
            self.do_check(trainer, train_metrics, epoch_idx, defer=True)

    def do_check(self, trainer, metrics, epoch_idx, defer=False):
        if self.monitor not in metrics:
            raise CallbackException(f'CheckCallback: {self.on_stage}阶段的指标中不包含 {self.monitor}!')
        decision = None
        if trainer.main_process:
            try:
                keep_training = self.check(trainer, metrics, trainer.model, trainer.opt, epoch_idx, defer=True)
                decision = (keep_training, self.best_value, self.early_stop_baseline,
                            self.worse_times, self.ckpt_path, self._should_save, self._save_path, None)
            except Exception as error:
                decision = (False, self.best_value, self.early_stop_baseline,
                            self.worse_times, self.ckpt_path, False, None, error)
        keep_training, self.best_value, self.early_stop_baseline, self.worse_times, self.ckpt_path, should_save, save_path, error = \
            trainer.distributed.broadcast_object(decision)
        if error is not None:
            raise error
        pending = getattr(trainer, '_pending_checkpoints', None)
        if pending is None:
            pending = trainer._pending_checkpoints = []
        pending.append((self, should_save, keep_training, epoch_idx, save_path))
        if not defer:
            finalize_checkpoints(trainer)


def finalize_checkpoints(trainer, callback_error=None):
    """在所有after_epoch回调完成后保存完整状态，再统一决定是否停止。"""
    pending = getattr(trainer, '_pending_checkpoints', [])
    if not pending:
        return
    trainer._pending_checkpoints = []
    decision = None
    callbacks = _check_callbacks(trainer)
    try:
        rank_states = _collect_rank_states(trainer) if any(item[1] for item in pending) else None
    except Exception as error:
        raise error from callback_error
    if trainer.main_process:
        try:
            # 快照记录所有计划成功的目标；实际路径只在对应文件写入成功后提交。
            planned_paths = {callbacks.index(callback): path
                             for callback, should_save, _, _, path in pending if should_save}
            for callback, should_save, _, epoch_idx, path in pending:
                if should_save:
                    save_state(trainer, trainer.model, trainer.opt, path, checkpoint_paths=planned_paths,
                               rank_states=rank_states, best_value=callback.best_value, worse_times=callback.worse_times,
                               epoch=epoch_idx)
                    callback.ckpt_path = path
                    callback._record_saved(trainer, epoch_idx)
            stopped = next((callback for callback, _, keep, _, _ in pending if not keep), None)
            message = None if stopped is None else (
                f'Early stopping triggered: {stopped.on_stage} {stopped.monitor}'
                f' | best={stopped.best_value:g} | patience={stopped.worse_times}/{stopped.patience}')
            decision = (message, None, [callback.ckpt_path for callback in callbacks])
        except Exception as error:
            decision = (None, error, [callback.ckpt_path for callback in callbacks])
    message, error, paths = trainer.distributed.broadcast_object(decision)
    for callback, path in zip(callbacks, paths):
        callback.ckpt_path = path
    if error is not None:
        raise error from callback_error
    # after_epoch可能由自定义回调主动停止；完成保存后保留该停止原因。
    if message is not None and callback_error is None:
        raise StopLoopException(message)


def restore_test_model(trainer):
    """Trainer.test的finally调用，测试失败时也恢复继续训练使用的权重。"""
    original = getattr(trainer, '_test_model_state', None)
    if original is not None:
        try:
            trainer.model.load_state_dict(original)
        finally:
            del trainer._test_model_state


def _read_state(trainer, ckpt_path):
    """读取当前格式的完整训练状态。"""
    return torch.load(ckpt_path, map_location='cpu', weights_only=False)


def _broadcast_state(trainer, ckpt_path):
    payload = None
    if trainer.main_process:
        try:
            payload = (_read_state(trainer, ckpt_path), None)
        except Exception as error:
            payload = (None, error)
    state, error = trainer.distributed.broadcast_object(payload)
    if error is not None:
        raise error
    return state


def _restore_training_state(trainer, state, ckpt_path, restore_model=True):
    checkpoint_file = state['checkpoint_file']
    callback_states = state['callback_states']
    if restore_model:
        trainer.model.load_state_dict(state['model_state'])
        trainer.opt.load_state_dict(state['opt_state'])
        if trainer.scaler.is_enabled() and state.get('scaler_state'):
            trainer.scaler.load_state_dict(state['scaler_state'])
        trainer.init_epoch = state['epoch'] + 1
    for ordinal, callback in enumerate(_check_callbacks(trainer)):
        saved = next((item for item in callback_states
                      if item['monitor'] == callback.monitor and item['on_stage'] == callback.on_stage
                      and item['ordinal'] == ordinal), None)
        if saved is not None:
            callback.load_state_dict(saved)
            relative = saved['checkpoint_path']
            if relative is None:
                callback.ckpt_path = None
            elif relative == checkpoint_file:
                callback.ckpt_path = osp.abspath(ckpt_path)
            else:
                callback.ckpt_path = osp.normpath(osp.join(osp.dirname(osp.abspath(ckpt_path)), relative))
        callback._resume_loaded = True


def load_state(trainer, ckpt_path):
    """立即恢复训练状态；回调、数据采样和随机数在下一次 fit 初始化后恢复。"""
    state = _broadcast_state(trainer, ckpt_path)
    _restore_training_state(trainer, state, ckpt_path)
    trainer._pending_fit_state = (state, osp.abspath(ckpt_path), False)
    return state


def _callback_identity(callback, ordinal):
    cls = type(callback)
    # spawn导入入口模块时使用__mp_main__，回调身份沿用脚本的主模块名。
    module = '__main__' if cls.__module__ == '__mp_main__' else cls.__module__
    return f'{module}.{cls.__qualname__}', getattr(callback, '_callback_ordinal', ordinal)


def _fit_loaders(trainer, train_tasks=None, val_tasks=None):
    stages = (('train', getattr(trainer, 'train_tasks', []) if train_tasks is None else train_tasks),
              ('val', getattr(trainer, 'val_tasks', []) if val_tasks is None else val_tasks))
    for stage, tasks in stages:
        for ordinal, task in enumerate(tasks):
            loader = getattr(task, f'_{stage}_dataloader', None)
            if loader is not None:
                yield stage, ordinal, loader


def _loader_state(loader):
    state = {}
    for name, target in (('loader', loader), ('sampler', getattr(loader, 'sampler', None)),
                         ('batch_sampler', getattr(loader, 'batch_sampler', None)),
                         ('batch_sampler_sampler', getattr(getattr(loader, 'batch_sampler', None), 'sampler', None))):
        values = {}
        generator = getattr(target, 'generator', None)
        if isinstance(generator, torch.Generator):
            values['generator'] = generator.get_state()
        if isinstance(target, (DistributedSampler, _DistributedEvalSampler)):
            values['seed'] = target.seed
        if values:
            state[name] = values
    return state


def _restore_loader_state(loader, state):
    targets = {'loader': loader, 'sampler': getattr(loader, 'sampler', None),
               'batch_sampler': getattr(loader, 'batch_sampler', None),
               'batch_sampler_sampler': getattr(getattr(loader, 'batch_sampler', None), 'sampler', None)}
    for name, values in state.items():
        target = targets[name]
        generator = getattr(target, 'generator', None)
        if 'generator' in values and isinstance(generator, torch.Generator):
            generator.set_state(values['generator'])
        if 'seed' in values and isinstance(target, (DistributedSampler, _DistributedEvalSampler)):
            target.seed = values['seed']


def _cpu_state(value):
    """回调显式状态使用独立CPU存储，避免对象通信携带设备或计算图。"""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_state(item) for item in value)
    return deepcopy(value)


def collect_rank_state(trainer):
    """采集本进程的 epoch 边界状态，不执行进程间通信。"""
    random_state = {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state()}
    if trainer.device.type == 'cuda':
        random_state['cuda'] = torch.cuda.get_rng_state(trainer.device)
    return {
        'random': random_state,
        'callbacks': [{'identity': _callback_identity(callback, ordinal),
                       'state': _cpu_state(callback.state_dict())}
                      for ordinal, callback in enumerate(trainer.callbacks)],
        'loaders': [{'stage': stage, 'ordinal': ordinal, 'state': _loader_state(loader)}
                    for stage, ordinal, loader in _fit_loaders(trainer)],
    }


def _collect_rank_states(trainer):
    try:
        payload = (collect_rank_state(trainer), None)
    except Exception as error:
        payload = (None, error)
    gathered = trainer.distributed.all_gather_object(payload)
    for _, error in gathered:
        if error is not None:
            raise error
    return [state for state, _ in gathered]


def prepare_fit_state(trainer):
    """在初始化回调前保存显式加载后的当前训练状态，包括调用方的修改。"""
    pending = getattr(trainer, '_pending_fit_state', None)
    if pending is None or pending[2]:
        return
    state, path, _ = pending
    # state_dict容器独立于模型；保留其元数据，仅将各字段复制到CPU存储。
    model_state = trainer.model.state_dict()
    for key, value in model_state.items():
        model_state[key] = _cpu_state(value)
    state = {**state,
             'model_state': model_state,
             'opt_state': _cpu_state(trainer.opt.state_dict()),
             'scaler_state': _cpu_state(trainer.scaler.state_dict()),
             'epoch': trainer.init_epoch - 1}
    trainer._pending_fit_state = (state, path, True)


def restore_fit_state(trainer, train_tasks, val_tasks):
    """在所有 before_fit 回调结束后，恢复待处理的 epoch 边界状态。"""
    pending = getattr(trainer, '_pending_fit_state', None)
    if pending is None:
        return
    state, path, restore_model = pending
    _restore_training_state(trainer, state, path, restore_model=restore_model)
    rank_states = state.get('rank_states', [])
    same_world_size = state.get('world_size') == trainer.world_size and len(rank_states) == trainer.world_size
    if rank_states:
        local = rank_states[trainer.rank] if same_world_size else rank_states[0]
        callback_states = {tuple(item['identity']): item['state'] for item in local['callbacks']}
        for ordinal, callback in enumerate(trainer.callbacks):
            saved = callback_states.get(_callback_identity(callback, ordinal))
            if saved is not None:
                callback.load_state_dict(saved)
        if same_world_size:
            loaders = {(item['stage'], item['ordinal']): item['state'] for item in local['loaders']}
            for stage, ordinal, loader in _fit_loaders(trainer, train_tasks, val_tasks):
                if (stage, ordinal) in loaders:
                    _restore_loader_state(loader, loaders[stage, ordinal])
            trainer._pending_fit_random_state = local['random']
    del trainer._pending_fit_state


def restore_fit_random_state(trainer):
    """在模型同步完成后恢复本进程 RNG，随后进入下一轮。"""
    state = getattr(trainer, '_pending_fit_random_state', None)
    if state is None:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and trainer.device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], trainer.device)
    del trainer._pending_fit_random_state


def load_evaluation_model(trainer, weights='auto'):
    """选择评估权重，保留当前模型供 restore_test_model 恢复。"""
    if weights == 'current':
        trainer.print(f"{_color('[WEIGHTS]', 'heading')} Using current model weights")
        return
    selected = None
    if weights == 'auto':
        selected = next((callback for callback in _check_callbacks(trainer) if callback.ckpt_path is not None), None)
        if selected is None:
            trainer.print(f"{_color('[WEIGHTS]', 'heading')} Using current model weights")
            return
        path = selected.ckpt_path
    elif isinstance(weights, CheckCallback):
        selected = weights
        path = weights.ckpt_path
        if path is None:
            raise ValueError('所选 CheckCallback 尚无 checkpoint。')
    elif isinstance(weights, (str, os.PathLike)):
        path = weights
    else:
        raise TypeError('weights 必须为 auto、current、CheckCallback 或 checkpoint 路径。')
    state = _broadcast_state(trainer, path)
    if getattr(trainer, '_test_model_state', None) is None:
        original = trainer.model.state_dict()
        for key, value in original.items():
            original[key] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        trainer._test_model_state = original
    trainer.model.load_state_dict(state['model_state'])
    monitor = '' if selected is None else f' | monitor: {selected.on_stage}/{selected.monitor}'
    epoch = state.get('epoch')
    saved_epoch = '' if epoch is None else f' | saved epoch: {epoch + 1}'
    trainer.print(f"{_color('[WEIGHTS]', 'heading')} Using checkpoint: {path}{monitor}{saved_epoch}")


def save_state(trainer, model, opt, path, checkpoint_paths=None, rank_states=None, **kwargs):
    """保存当前模型、优化器、scaler和本轮所有监控器状态。"""
    if not trainer.main_process:
        return
    check_path(osp.dirname(osp.abspath(path)))
    state = {'model_state': model.state_dict(), 'opt_state': opt.state_dict(), **kwargs}
    state['world_size'] = trainer.world_size
    state['rank_states'] = [collect_rank_state(trainer)] if rank_states is None else rank_states
    snapshot_dir = osp.dirname(osp.abspath(path))
    state['checkpoint_file'] = osp.basename(path)
    state['callback_states'] = []
    for ordinal, callback in enumerate(_check_callbacks(trainer)):
        reference = (checkpoint_paths or {}).get(ordinal, callback.ckpt_path)
        state['callback_states'].append({
            'ordinal': ordinal, 'monitor': callback.monitor, 'on_stage': callback.on_stage,
            'best_value': callback.best_value, 'worse_times': callback.worse_times,
            'early_stop_baseline': callback.early_stop_baseline,
            'checkpoint_path': osp.relpath(reference, snapshot_dir) if reference is not None else None,
        })
    if trainer.scaler.is_enabled():
        state['scaler_state'] = trainer.scaler.state_dict()
    torch.save(state, path)


def _checkpoint_paths(run_dir):
    if not osp.isdir(run_dir):
        return []
    paths = []
    for name in os.listdir(run_dir):
        if name != 'checkpoint.ckpt' and not (name.startswith('checkpoint-') and name.endswith('.ckpt')):
            continue
        path = osp.join(run_dir, name)
        if osp.isfile(path):
            paths.append(path)
    return paths


def get_latest_running(from_dir):
    if not osp.isdir(from_dir):
        return None
    runs = [(max(osp.getmtime(path) for path in paths), name)
            for name in os.listdir(from_dir)
            if (paths := _checkpoint_paths(osp.join(from_dir, name)))]
    return max(runs)[1] if runs else None
