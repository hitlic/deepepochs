from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from os import path as osp
from ..tools import plot_confusion, TopKQueue
from ..metrics import confusion_matrix, get_class_num
from ..loops import check_path
from .callback import Callback
from .log import run_tensorboard


class InterpreteCallback(Callback):
    def __init__(self, metric=None, k=100, mode='max', stages=('train', 'val', 'test'), class_num=None, log_dir='./logs', image_data=False):
        """
        Args:
            metric:      none reducted callable
            k:           number of samples to keep
            mode:        'max' or 'min'
            stages:      'train' 'val' or 'test'
            class_num:   类别数量
            log_dir:     日志存储路径
            image_data:  数据是否是图片（如果是则在tensorboard中保存图片）
        """
        super().__init__()
        assert mode in ['min', 'max']
        stages = stages if isinstance(stages, (list, tuple)) else [stages]
        assert all(s in ['train', 'val', 'test'] for s in stages ), 'stages的值为 train、val、test或者其组合'
        self.metric = metric
        self.stages = stages
        self.mode = mode
        self.batch_recorder = []
        self.top_queue = TopKQueue(k=k)
        self.confusion_matrix = None
        self.class_num = class_num
        self.log_dir = log_dir
        self.image_data = image_data

    def on_before_fit(self, trainer, epochs):
        log_dir = osp.join(self.log_dir, trainer.running_id)
        check_path(log_dir)
        logger = getattr(trainer, 'logger', None)
        if logger is None:
            self.logger = SummaryWriter(log_dir=log_dir)
            trainer.logger = self.logger
        else:
            self.logger = logger

    def on_before_train_epochs(self, trainer, tasks, epoch_idx):
        self.confusion_matrix=None

    def on_before_val_epochs(self, trainer, tasks, epoch_idx):
        self.confusion_matrix=None

    def on_before_test_epochs(self, trainer, tasks):
        self.confusion_matrix=None

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        if self.class_num is None:
            self.class_num = get_class_num(batch_x, batch_y)
        self.batch_x = batch_x
        self.batch_y = batch_y

    def on_before_val_batch(self, trainer, batch_x, batch_y, batch_idx):
        self.batch_x = batch_x
        self.batch_y = batch_y

    def on_before_test_batch(self, trainer, batch_x, batch_y, batch_idx):
        self.batch_x = batch_x
        self.batch_y = batch_y

    def update_confusion_matrix(self, model_out):
        """更新混淆矩阵"""
        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix(model_out, self.batch_y, self.class_num)
        else:
            self.confusion_matrix += confusion_matrix(model_out, self.batch_y, self.class_num)

    def on_after_train_loss(self, trainer, loss, model_out, targets, task):
        if 'train' in self.stages:
            self.update_confusion_matrix(model_out)

    def on_after_val_loss(self, trainer, loss, model_out, targets, task):
        if 'val' in self.stages:
            self.update_confusion_matrix(model_out)

    def on_after_test_loss(self, trainer, loss, model_out, targets, task):
        if 'test' in self.stages:
            self.put_queue(model_out, self.batch_y, self.batch_x)
            self.update_confusion_matrix(model_out)

    def on_after_train_epochs(self, trainer, tasks, metrics, epoch_idx):
        if 'train' in self.stages:
            fig = self.plot_confusion(show=False)
            self.logger.add_figure('train_confusion_matrixes', fig, epoch_idx)
            plt.close(fig)

    def on_after_val_epochs(self, trainer, tasks, metrics, epoch_idx):
        if 'val' in self.stages:
            fig = self.plot_confusion(show=False)
            self.logger.add_figure('val_confusion_matrixes', fig, epoch_idx)
            plt.close(fig)

    def on_after_test_epochs(self, trainer, tasks, metrics):
        if 'test' in self.stages:
            if self.image_data:  # 在tensorboard中保存图像数据
                for i, data in enumerate(self.top_queue.queue):
                    self.logger.add_image('top_imgs', data[1][3], i)
            fig = self.plot_confusion(show=False)
            self.logger.add_figure('test_confusion_matrix', fig)
            plt.close(fig)

    def put_queue(self, preds, targets, inputs):
        if self.metric is None:
            return
        batch_m = self.metric(preds, targets)
        assert batch_m.shape[0] == preds.shape[0], 'The `metric` must not be reduced!'
        for m, pred, target, feat in zip(batch_m.cpu(), preds.cpu(), targets.cpu(), inputs[0].cpu()):
            v = -m if self.mode == 'min' else m
            self.top_queue.put((v.item(), [m.item(), pred, target, feat]))

    def top_samples(self):
        """
        Return: [metric_value, pred, target, input_feat] of top k samples 
        """
        samples = [item[1] for item in self.top_queue.items()]
        return samples

    def top(self):
        return self.top_samples()

    def plot_confusion(self, class_names=None, cmap='Blues', title_info='', show=True):
        c_matrix = self.confusion_matrix
        fig = plot_confusion(c_matrix.cpu().numpy(), self.class_num, class_names, cmap=cmap, info=title_info)
        if show:
            plt.show()
        return fig

    def run_tensorboard(self):
        run_tensorboard(self.log_dir)
