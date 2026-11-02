from os import path as osp

from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from ..loops import check_path
from ..metrics import confusion_matrix, get_class_num
from ..tools import TopKQueue, plot_confusion
from .callback import Callback
from .log import run_tensorboard


def _sample_data(data, index):
    if isinstance(data, dict):
        return {key: _sample_data(value, index) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return tuple(_sample_data(value, index) for value in data)
    # CPU 切片也会引用整批 storage；候选样本必须拥有独立的小块存储，
    # 才能让 TopK 内存和进程间序列化量随保留样本数增长。
    return data[index].detach().cpu().clone() if isinstance(data, Tensor) else data


def _image_input(inputs):
    while isinstance(inputs, (dict, list, tuple)):
        inputs = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs[0]
    return inputs


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
            image_data:  在tensorboard中保存图片；多输入使用第一个输入，字典按插入顺序
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
        self.prepare_logger(trainer)

    def prepare_logger(self, trainer):
        if not trainer.main_process:
            return
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
        self.prepare_logger(trainer)
        self.confusion_matrix=None
        self.top_queue = TopKQueue(self.top_queue.k)

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
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
        if self.class_num is None:
            self.class_num = get_class_num(model_out, self.batch_y)
        batch_matrix = confusion_matrix(model_out, self.batch_y, self.class_num).detach().clone()
        if self.confusion_matrix is None:
            self.confusion_matrix = batch_matrix
        else:
            self.confusion_matrix = self.confusion_matrix + batch_matrix

    def gather_confusion_matrix(self, trainer):
        context = getattr(trainer, 'distributed', None)
        if context is not None and context.is_distributed:
            local = None if self.confusion_matrix is None else self.confusion_matrix.cpu()
            matrices = [matrix for matrix in context.all_gather_object(local) if matrix is not None]
            self.confusion_matrix = sum(matrices) if matrices else None
            if self.confusion_matrix is not None:
                self.class_num = self.confusion_matrix.shape[0]

    def on_train_metrics(self, trainer, loss, model_out, batch_y, task):
        if 'train' in self.stages:
            self.batch_y = batch_y
            self.update_confusion_matrix(model_out)

    def on_val_metrics(self, trainer, loss, model_out, batch_y, task):
        if 'val' in self.stages:
            self.batch_y = batch_y
            self.update_confusion_matrix(model_out)

    def on_test_metrics(self, trainer, loss, model_out, batch_y, task):
        if 'test' in self.stages:
            self.batch_y = batch_y
            self.put_queue(model_out, self.batch_y, self.batch_x)
            self.update_confusion_matrix(model_out)

    def on_after_train_epochs(self, trainer, tasks, metrics, epoch_idx):
        if 'train' in self.stages:
            self.gather_confusion_matrix(trainer)
            if not trainer.main_process or self.confusion_matrix is None:
                return
            fig = self.plot_confusion(show=False)
            self.logger.add_figure('train_confusion_matrixes', fig, epoch_idx)
            plt.close(fig)

    def on_after_val_epochs(self, trainer, tasks, metrics, epoch_idx):
        if 'val' in self.stages:
            self.gather_confusion_matrix(trainer)
            if not trainer.main_process or self.confusion_matrix is None:
                return
            fig = self.plot_confusion(show=False)
            self.logger.add_figure('val_confusion_matrixes', fig, epoch_idx)
            plt.close(fig)

    def on_after_test_epochs(self, trainer, tasks, metrics):
        if 'test' in self.stages:
            self.gather_confusion_matrix(trainer)
            context = getattr(trainer, 'distributed', None)
            if context is not None and context.is_distributed:
                candidates = context.all_gather_object(self.top_queue.items())
                if trainer.main_process:
                    self.top_queue = TopKQueue(self.top_queue.k)
                    for rank_candidates in candidates:
                        for item in rank_candidates:
                            self.top_queue.put(item)
            if not trainer.main_process:
                return
            if self.image_data:  # 在tensorboard中保存图像数据
                for i, data in enumerate(self.top_queue.items()):
                    self.logger.add_image('top_imgs', _image_input(data[1][3]), i)
            if self.confusion_matrix is not None:
                fig = self.plot_confusion(show=False)
                self.logger.add_figure('test_confusion_matrix', fig)
                plt.close(fig)

    def put_queue(self, preds, targets, inputs):
        if self.metric is None:
            return
        batch_m = self.metric(preds, targets)
        assert batch_m.shape[0] == preds.shape[0], 'The `metric` must not be reduced!'
        for index, metric in enumerate(batch_m.detach().cpu()):
            metric_value = metric.item()
            score = -metric_value if self.mode == 'min' else metric_value
            if len(self.top_queue.heap) == self.top_queue.k and score <= self.top_queue.heap[0][0]:
                continue
            pred = _sample_data(preds, index)
            target = _sample_data(targets, index)
            feat = _sample_data(inputs, index)
            self.top_queue.put((score, [metric_value, pred, target, feat]))

    def top_samples(self):
        """
        Return: [metric_value, pred, target, input_feat] of top k samples.
        input_feat is a tensor for one input, a tuple or dict for multiple inputs.
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
