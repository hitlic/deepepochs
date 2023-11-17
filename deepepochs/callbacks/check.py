
from .callback import Callback, CallbackException
import torch
import os
from os import path as osp
from ..loops import check_path, StopLoopException


class CheckCallback(Callback):
    def __init__(self, monitor, on_stage='val', mode='min', patience=0, ckpt_dir='./logs'):
        """
        实现功能：Checkpoint和Early Stop。其中，仅保存监控指标最优Checkpoint。
        Args:
            monitor:   监控指标
            on_stage:  监控目标，'train'或'val'
            mode:      监控指标模式，'max'或'min'
            patience:  Early Stop 容忍指标连续变差的次数，0表示不启用Early Stop
            ckpt_dir:  最优模型Checkpoint的保存位置
        """
        self.monitor = monitor
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

    def check(self, metrics, model, opt):
        """
        Reture:
            True:  表示继续执行
            False: 表示达到Early Stop条件
        """
        value = metrics[self.monitor]
        if self.mode == 'max':
            if  value > self.best_value:
                self.best_value = value
                save_state(model, opt, self.ckpt_path, best_value=self.best_value)
                self.worse_times = 0
            else:
                self.worse_times += 1
        else:
            if value < self.best_value:
                self.best_value = value
                save_state(model, opt, self.ckpt_path, best_value=self.best_value)
                self.worse_times = 0
            else:
                self.worse_times += 1
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
                print(f'loading checkpoint of running {running_id} ...')
                path = osp.join(self.ckpt_dir, running_id, 'checkpoint.ckpt')
                self.load_state(trainer, path)
            except FileNotFoundError:
                print('loading failed, checkpoint does not exist!\nstarting training with random parameters!')
            except Exception as e:
                print(f'loading failed! {e}\nstarting training with random parameters!')

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        if self.on_stage == 'val':
            if val_tasks and (epoch_idx+1)%val_tasks[0].val_freq==0:
                self.do_check(trainer, val_metrics)
        else:
            self.do_check(trainer, train_metrics)

    def do_check(self, trainer, metrics):
        if self.monitor in metrics:
            # 创建新的checkpoint路径
            ckpt_dir = osp.join(self.ckpt_dir, trainer.running_id)
            check_path(ckpt_dir)
            self.ckpt_path = osp.join(ckpt_dir, 'checkpoint.ckpt')
            # 检查（保存checkpoint，early stop）
            if not self.check(metrics, trainer.model, trainer.opt):
                raise StopLoopException(f"Early stopping triggered, by monitoring [{self.on_stage} {self.monitor}]!")
        else:
            raise CallbackException(f"CheckCallback: {self.on_stage}阶段的指标中不包含 {self.monitor}!")

    def on_before_test_epochs(self, trainer, tasks):
        try:
            if self.ckpt_path is not None:
                print(f'loading best model from running {trainer.running_id} ...')
                self.load_state(trainer, self.ckpt_path)
        except FileNotFoundError as e:
            print('loading best failed,', e)
            print('testing with leatest model.')

    def load_state(self, trainer, ckpt_path):
        other_params = load_state(trainer.model, trainer.opt, ckpt_path)
        for k, v in other_params.items():
            setattr(self, k, v)


def save_state(model, opt, path, **kwargs):
    state = {'model_state': model.state_dict(), 'opt_state': opt.state_dict(), **kwargs}
    torch.save(state, path)


def load_state(model, opt, path):
    state = torch.load(path)
    model.load_state_dict(state['model_state'])
    opt.load_state_dict(state['opt_state'])
    return {k: v for k, v in state.items() if k not in ['model_state', 'opt_state']}


def get_latest_running(from_dir):
    try:
        dir_list = [f for f in os.listdir(from_dir) if osp.isdir(osp.join(from_dir, f))]
        file_list = sorted(dir_list, key=lambda f: osp.getctime(osp.join(from_dir, f)))
        return file_list[-1]
    except Exception:
        return 'NULL_checkpoint'
