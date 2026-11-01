import os
from copy import deepcopy
from os import path as osp

import torch

from ..loops import StopLoopException, check_path
from .callback import Callback, CallbackException


def _check_callbacks(trainer):
    callbacks = [callback for callback in trainer.callbacks if isinstance(callback, CheckCallback)]
    return sorted(callbacks, key=lambda callback: getattr(callback, '_checkpoint_ordinal', 0))


def _checkpoint_path(trainer, callback):
    callbacks = _check_callbacks(trainer)
    name = 'checkpoint.ckpt' if len(callbacks) == 1 else f'checkpoint-{callbacks.index(callback) + 1}.ckpt'
    return osp.join(callback.ckpt_dir, trainer.running_id, name)


class CheckCallback(Callback):
    def __init__(self, monitor, on_stage='val', mode='min', patience=0, save_best=True, ckpt_dir='./logs'):
        """保存监控指标对应的Checkpoint，并按连续未改善的轮数早停。

        多个CheckCallback分别保存checkpoint-1.ckpt、checkpoint-2.ckpt；
        只有一个时使用checkpoint.ckpt。测试临时使用最佳模型，结束后恢复当前模型。

        Args:
            monitor: 指标名；on_stage指定train或val。
            mode: min或max，决定改善方向。
            patience: 连续未改善多少轮后停止，0表示不启用早停。
            save_best: True保存最佳，False保存最近一轮。
            ckpt_dir: 各次训练Checkpoint的父目录。
        """
        assert on_stage in ['train', 'val'], 'CheckCallback的`on_stage`参数取值为"train"或"val"'
        assert mode in ['min', 'max']
        self.monitor = monitor
        self.on_stage = on_stage
        self.mode = mode
        self.patience = patience
        self.save_best = save_best
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = None
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.worse_times = 0
        super().__init__(priority=-1)

    def check(self, trainer, metrics, model, opt, epoch_idx, defer=False):
        """更新监控状态；defer用于等待本轮其他回调更新后统一保存。"""
        value = float(metrics[self.monitor])
        better = value > self.best_value if self.mode == 'max' else value < self.best_value
        if better:
            self.best_value = value
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
        return not (self.patience > 0 and self.worse_times >= self.patience)

    def on_before_fit(self, trainer, epochs):
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
            trainer.print('No checkpoint found; starting a new training run.')
            for callback in callbacks:
                callback._resume_loaded = True
            return
        trainer.print(f'loading checkpoint {path} ...')
        load_state(trainer, path)

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
                decision = (keep_training, self.best_value, self.worse_times, self.ckpt_path,
                            self._should_save, self._save_path, None)
            except Exception as error:
                decision = (False, self.best_value, self.worse_times, self.ckpt_path, False, None, error)
        keep_training, self.best_value, self.worse_times, self.ckpt_path, should_save, save_path, error = \
            trainer.distributed.broadcast_object(decision)
        if error is not None:
            raise error
        pending = getattr(trainer, '_pending_checkpoints', None)
        if pending is None:
            pending = trainer._pending_checkpoints = []
        pending.append((self, should_save, keep_training, epoch_idx, save_path))
        if not defer:
            finalize_checkpoints(trainer)

    def on_before_test_epochs(self, trainer, tasks):
        if self.ckpt_path is None or getattr(trainer, '_test_model_state', None) is not None:
            return
        selected = next(callback for callback in _check_callbacks(trainer) if callback.ckpt_path is not None)
        if selected is not self:
            return
        trainer.print(f'loading best model from {self.ckpt_path} ...')
        state = _broadcast_state(trainer, self.ckpt_path)
        # state_dict是独立映射；复制到CPU保留参数和buffer，同时不多占一份GPU模型。
        original = trainer.model.state_dict()
        for key, value in original.items():
            original[key] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        trainer._test_model_state = original
        trainer.model.load_state_dict(state['model_state'])


def finalize_checkpoints(trainer, callback_error=None):
    """在所有after_epoch回调完成后保存完整状态，再统一决定是否停止。"""
    pending = getattr(trainer, '_pending_checkpoints', [])
    if not pending:
        return
    trainer._pending_checkpoints = []
    decision = None
    callbacks = _check_callbacks(trainer)
    if trainer.main_process:
        try:
            # 快照记录所有计划成功的目标；实际路径只在对应文件写入成功后提交。
            planned_paths = {callbacks.index(callback): path
                             for callback, should_save, _, _, path in pending if should_save}
            for callback, should_save, _, epoch_idx, path in pending:
                if should_save:
                    save_state(trainer, trainer.model, trainer.opt, path, checkpoint_paths=planned_paths,
                               best_value=callback.best_value, worse_times=callback.worse_times,
                               epoch=epoch_idx)
                    callback.ckpt_path = path
            stopped = next((callback for callback, _, keep, _, _ in pending if not keep), None)
            message = None if stopped is None else \
                f'Early stopping triggered, by monitoring [{stopped.on_stage} {stopped.monitor}]!'
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


def load_state(trainer, ckpt_path):
    """主进程读取训练状态并广播，各进程恢复到相同的下一轮。"""
    state = _broadcast_state(trainer, ckpt_path)
    checkpoint_file = state['checkpoint_file']
    callback_states = state['callback_states']
    actual_path = osp.abspath(ckpt_path)
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
            callback.best_value = saved['best_value']
            callback.worse_times = saved['worse_times']
            relative = saved['checkpoint_path']
            if relative is None:
                callback.ckpt_path = None
            elif relative == checkpoint_file:
                # 当前文件可以改名；兄弟监控器仍指向各自的最佳文件。
                callback.ckpt_path = actual_path
            else:
                callback.ckpt_path = osp.normpath(osp.join(osp.dirname(actual_path), relative))
        callback._resume_loaded = True
    return state


def save_state(trainer, model, opt, path, checkpoint_paths=None, **kwargs):
    """保存当前模型、优化器、scaler和本轮所有监控器状态。"""
    if not trainer.main_process:
        return
    check_path(osp.dirname(osp.abspath(path)))
    state = {'model_state': model.state_dict(), 'opt_state': opt.state_dict(), **kwargs}
    snapshot_dir = osp.dirname(osp.abspath(path))
    state['checkpoint_file'] = osp.basename(path)
    state['callback_states'] = []
    for ordinal, callback in enumerate(_check_callbacks(trainer)):
        reference = (checkpoint_paths or {}).get(ordinal, callback.ckpt_path)
        state['callback_states'].append({
            'ordinal': ordinal, 'monitor': callback.monitor, 'on_stage': callback.on_stage,
            'best_value': callback.best_value, 'worse_times': callback.worse_times,
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
