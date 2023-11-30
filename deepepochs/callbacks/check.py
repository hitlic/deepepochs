
from .callback import Callback, CallbackException
import torch
import os
from os import path as osp
from ..loops import check_path, StopLoopException


class CheckCallback(Callback):
    def __init__(self, monitor, on_stage='val', mode='min', patience=0, save_best=True, ckpt_dir='./logs'):
        """
        实现功能：Checkpoint和Early Stop。其中，仅保存监控指标最优Checkpoint。
        Args:
            monitor:   监控指标
            on_stage:  监控目标，'train'或'val'
            mode:      监控指标模式，'max'或'min'
            patience:  Early Stop 容忍指标连续变差的次数，0表示不启用Early Stop
            save_best: True保存最佳Checkpoint，False保存最新Checkpoint
            ckpt_dir:  最优模型Checkpoint的保存位置
        """
        self.monitor = monitor
        self.save_best = save_best
        assert on_stage in ['train', 'val'], 'CheckCallback的`on_stage`参数取值为"train"或"val"'
        self.on_stage = on_stage
        self.mode = mode
        self.patience = patience

        assert mode in ['min', 'max']
        self.best_value = -100000000.0 if  mode == 'max' else 100000000.0

        self.ckpt_dir = ckpt_dir
        self.ckpt_path = None
        self.worse_times = 0
        super().__init__(priority=-1)

    def check(self, trainer, metrics, model, opt, epoch_idx):
        """
        Reture:
            True:  表示继续执行
            False: 表示达到Early Stop条件
        """
        value = metrics[self.monitor]
        if not self.save_best:      # 保存最新Checkpoint
            save_state(trainer, model, opt, self.ckpt_path, best_value=self.best_value, epoch=epoch_idx)

        if self.mode == 'max':
            better = value > self.best_value
        else:
            better = value < self.best_value

        if better:
            self.best_value = value
            self.worse_times = 0
            if self.save_best:          # 保存最佳Checkpoint
                save_state(trainer, model, opt, self.ckpt_path, best_value=self.best_value, epoch=epoch_idx)
        else:
            self.worse_times += 1

        # Early Stopping
        if self.patience > 0 and self.worse_times >= self.patience:
            return False
        return True

    def on_before_fit(self, trainer, epochs):
        if trainer.resume is not False:
            if trainer.resume is True:
                running_id = get_latest_running(self.ckpt_dir)  # 加载最近的checkpoint
            else:
                running_id = str(trainer.resume)                # 加载指定的checkpoint
            try:
                trainer.print(f'loading checkpoint of running {running_id} ...')
                path = osp.join(self.ckpt_dir, running_id, 'checkpoint.ckpt')
                load_state(trainer, path)
            except FileNotFoundError:
                trainer.print('loading failed, checkpoint does not exist!\nstarting training with random parameters!')
            except Exception as e:
                trainer.print(f'loading failed! {e}\nstarting training with random parameters!')

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        if self.on_stage == 'val':
            if val_tasks and (epoch_idx+1)%val_tasks[0].val_freq==0:
                self.do_check(trainer, val_metrics, epoch_idx)
        else:
            self.do_check(trainer, train_metrics, epoch_idx)

    def do_check(self, trainer, metrics, epoch_idx):
        if self.monitor in metrics:
            # 创建新的checkpoint路径
            ckpt_dir = osp.join(self.ckpt_dir, trainer.running_id)
            check_path(ckpt_dir)
            self.ckpt_path = osp.join(ckpt_dir, 'checkpoint.ckpt')
            # 检查（保存checkpoint，early stop）
            if not self.check(trainer, metrics, trainer.model, trainer.opt, epoch_idx):
                raise StopLoopException(f"Early stopping triggered, by monitoring [{self.on_stage} {self.monitor}]!")
        else:
            raise CallbackException(f"CheckCallback: {self.on_stage}阶段的指标中不包含 {self.monitor}!")

    def on_before_test_epochs(self, trainer, tasks):
        try:
            if self.ckpt_path is not None:
                trainer.print(f'loading best model from running {trainer.running_id} ...')
                load_state(trainer, self.ckpt_path)
        except FileNotFoundError as e:
            trainer.print('loading best failed,', e)
            trainer.print('testing with leatest model.')


def load_state(trainer, ckpt_path):
    if trainer.accelerator is None:
        state = torch.load(ckpt_path)
        trainer.model.load_state_dict(state['model_state'])
        trainer.opt.load_state_dict(state['opt_state'])
        epochs = state['epoch']
    else:
        trainer.accelerator.load_state(ckpt_path)
        epochs = torch.load(osp.join(ckpt_path, 'meta.ckpt'))['epoch']
    trainer.init_epoch = epochs + 1


def save_state(trainer, model, opt, path, **kwargs):
    if trainer.accelerator is None:
        model_state = model.state_dict()
        state = {'model_state': model_state,'opt_state': opt.state_dict(), **kwargs}
        torch.save(state, path)
    else:
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.save_state(path)
        torch.save(kwargs, osp.join(path, 'meta.ckpt'))


def get_latest_running(from_dir):
    try:
        dir_list = [f for f in os.listdir(from_dir) if osp.isdir(osp.join(from_dir, f))]
        file_list = sorted(dir_list, key=lambda f: osp.getctime(osp.join(from_dir, f)))
        return file_list[-1]
    except Exception:
        return 'NULL_checkpoint'
