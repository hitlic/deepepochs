import torch
from .callback import Callback
from ..loops import log_batch, log_epoch, batch_size
from ..patches import ValuePatch


class DefaultCallback(Callback):
    def __init__(self, log_long, log_batch, log_tqdm):
        """
        默认启用的Callback，实现功能：
            指标输出
            学习率调度
            为mini-batch构建每个指标的Patch
        Args:
            log_long:  指标输出为长格式（7位小说）还是短格式（4位小数）
            log_batch: 是否输出batch的指标值
            tqdm_iter: tqdm迭代对象
        """
        super().__init__(priority=0)
        self.round_to = 7 if log_long else 4
        self.log_batch = log_batch
        self.epoch_width = 4
        self.batch_width = 5
        self.log_tqdm = log_tqdm
        self.tqdm_iter = None

    def on_before_fit(self, trainer, epochs):
        self.total_epochs = epochs

    def on_before_epoch(self, trainer, train_tasks, val_tasks, epoch_idx):
        self.epoch_idx = epoch_idx
        self.total_train_batchs = sum(task.batchs for task in train_tasks)  # 所有训练任务总batch数量
        self.total_val_batchs = sum(task.batchs for task in val_tasks)      # 所有验证任务总batch数量
        self.global_train_batch_idx = 0                                     # 当前训练batch
        self.global_val_batch_idx = 0                                       # 当前验证batch

        self.epoch_width = len(str(self.total_epochs))
        self.batch_width = len(str(max(self.total_val_batchs, self.total_train_batchs)))

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.global_train_batch_idx += 1
        if self.log_batch and trainer.main_process:
            log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.global_train_batch_idx, self.total_train_batchs, 'TRAIN', self.epoch_width, self.batch_width, self.round_to)

    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.global_val_batch_idx += 1
        if self.log_batch and trainer.main_process:
            log_batch(metrics, self.epoch_idx+1, self.total_epochs, self.global_val_batch_idx, self.total_val_batchs, 'VAL', self.epoch_width, self.batch_width, self.round_to)

    def on_after_epoch(self, trainer, train_tasks, val_tasks, train_metrics, val_metrics, epoch_idx):
        if trainer.main_process:
            if val_metrics:
                log_epoch({'train': train_metrics, 'val': val_metrics}, epoch_idx+1, self.total_epochs, self.epoch_width, self.round_to, self.tqdm_iter)
            else:
                log_epoch({'train': train_metrics}, epoch_idx+1, self.total_epochs, self.epoch_width, self.round_to, self.tqdm_iter)

        # 根据调度器的配置改变优化器学习率
        if val_metrics:  # 优先使用验证损失
            sched_loss = val_metrics.get('loss')
        else:
            sched_loss = train_metrics.get('loss')
        trainer.opt.step(at='epoch', loss=sched_loss)

    def on_before_test_epochs(self, trainer, tasks):
        self.total_test_epochs = len(tasks)
        self.global_test_epoch_idx = 0
        self.total_test_batchs = sum(task.batchs for task in tasks)
        self.global_test_batch_idx = 0

    def on_after_test_epoch(self, trainer, task, metrics):
        if trainer.main_process:
            log_epoch({'test': metrics}, self.global_test_epoch_idx+1, self.total_test_epochs, self.epoch_width, self.round_to)
        self.global_test_epoch_idx += 1

    def on_after_test_batch(self, trainer, metrics, batch_idx):
        self.global_test_batch_idx += 1
        if self.log_batch and trainer.main_process:
            log_batch(metrics, self.global_test_epoch_idx+1, self.total_test_epochs, self.global_test_batch_idx, self.total_test_batchs, 'TEST', self.epoch_width, self.batch_width, self.round_to)

    def on_train_metrics(self, trainer, loss, model_out, batch_y, task):
        """当前task的每个指标构建Patch，并注入task.batch_patch_dict"""
        task.batch_patch_dict = self.make_patch_dict(trainer, loss, model_out, batch_y, task.metrics, 'train')

    def on_val_metrics(self, trainer, loss, model_out, batch_y, task):
        """当前task的每个指标构建Patch，并注入task.batch_patch_dict"""
        task.batch_patch_dict = self.make_patch_dict(trainer, loss, model_out, batch_y, task.metrics, 'val')

    def on_test_metrics(self, trainer, loss, model_out, batch_y, task):
        """当前task的每个指标构建Patch，并注入task.batch_patch_dict"""
        task.batch_patch_dict = self.make_patch_dict(trainer, loss, model_out, batch_y, task.metrics, 'test')

    def make_patch_dict(self, trainer, loss, model_out, batch_y, metrics, stage):
        b_size = torch.tensor(batch_size(model_out)).to(trainer.device)
        # Accelerate 分布式训练时，获取各Process的数据
        if trainer.accelerator is not None and stage!='train':  # 训练时仅在主线程上计算指标
            if loss is not None:
                loss = trainer.accelerator.gather_for_metrics(loss)
                b_size = trainer.accelerator.gather_for_metrics(b_size)
                loss = (loss * b_size).sum()
                b_size = b_size.sum()
                loss = loss/b_size
            model_out = trainer.accelerator.gather_for_metrics(model_out)
            batch_y = trainer.accelerator.gather_for_metrics(batch_y)

        patch_dict = {} if loss is None else  {'loss': ValuePatch(loss, b_size)}
        for m in metrics:
            patch_dict[m.__name__] = trainer.metric_patch(m, model_out, batch_y)
        return patch_dict
