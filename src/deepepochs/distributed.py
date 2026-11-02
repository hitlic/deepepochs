"""单进程与 PyTorch DDP 共用的设备、数据分片和通信入口。"""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class _DistributedEvalSampler(Sampler):
    """验证时不补齐样本；数据较少时，部分进程可以没有样本。"""

    def __init__(self, dataset, rank, world_size, shuffle=False, seed=0, sample_count=None):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.sample_count = len(dataset) if sample_count is None else sample_count

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = range(len(self.dataset))
        # drop_last 针对原 loader 选择完整批量，再分片；不能分片后各自丢尾，
        # 否则原本足够构成一批的数据也可能在所有进程上同时被丢掉。
        indices = indices[:self.sample_count]
        return iter(indices[self.rank::self.world_size])

    def __len__(self):
        return len(range(self.rank, self.sample_count, self.world_size))

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedContext:
    """通过 torchrun 环境启用 DDP，普通 Python 启动保持单进程行为。

    已存在的进程组归调用方管理，close 不会销毁它。CUDA 通信前设置当前
    设备，NCCL 的对象通信也依赖这个设备，即使传递的数据本身在 CPU 上。
    """

    def __init__(self, device=None):
        self._owns_process_group = False
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        initialized = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if initialized else int(os.environ.get('WORLD_SIZE', 1))

        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type == 'cuda':
            index = self.device.index if self.device.index is not None else self.local_rank
            self.device = torch.device('cuda', index)
            torch.cuda.set_device(self.device)

        if world_size > 1 and not initialized:
            if self.device.type not in ('cpu', 'cuda'):
                raise ValueError('PyTorch DDP 需要 CPU 或 CUDA 设备。')
            backend = 'nccl' if self.device.type == 'cuda' else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://')
            self._owns_process_group = True
            initialized = True

        self.rank = dist.get_rank() if initialized else 0
        self.world_size = dist.get_world_size() if initialized else 1
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

    def wrap_model(self, model, **ddp_kwargs):
        """先放到当前设备，再封装 DDP；调用方可传 DDP 参数。"""
        if isinstance(model, DistributedDataParallel):
            return model
        model = model.to(self.device)
        if not self.is_distributed:
            return model
        if self.device.type == 'cuda':
            ddp_kwargs.setdefault('device_ids', [self.device.index])
            ddp_kwargs.setdefault('output_device', self.device.index)
        return DistributedDataParallel(model, **ddp_kwargs)

    def prepare_dataloader(self, loader, training=True, split_batches=False):
        """分片普通 DataLoader，不替换自定义采样逻辑。

        默认 batch_size 是每进程批量；split_batches=True 将原批量平均分给
        各进程，要求可以整除。单进程和已有 DistributedSampler 的 loader
        原样返回。自定义 sampler、batch_sampler 或 DataLoader 子类也原样
        返回，其分布式分片由调用方管理。

        训练使用等长分片；drop_last=False 时 DistributedSampler 会补齐样本。
        验证不补齐，各进程长度可以不同，调用方应使用原始模型验证，并在
        整个任务结束后通信。验证的 drop_last 先按原批量选择样本，再分片，
        因而评估的样本集合与原 loader 一致，不在各进程重复丢尾。
        """
        if not self.is_distributed or type(loader) is not DataLoader:
            return loader
        if isinstance(loader.sampler, DistributedSampler):
            return loader
        if type(loader.sampler) not in (RandomSampler, SequentialSampler):
            return loader
        if type(loader.batch_sampler) is not BatchSampler or loader.batch_size is None:
            return loader
        if isinstance(loader.sampler, RandomSampler):
            # replacement/num_samples/独立 generator 都属于调用方指定的采样规则。
            if loader.sampler.replacement or loader.sampler.num_samples != len(loader.dataset):
                return loader
            if loader.sampler.generator is not loader.generator:
                return loader

        batch_size = loader.batch_size
        if split_batches:
            if batch_size % self.world_size:
                raise ValueError('split_batches=True 时 batch_size 必须能被进程数量整除。')
            batch_size //= self.world_size

        shuffle = isinstance(loader.sampler, RandomSampler)
        # 所有进程必须使用同一个排列，再分别取分片。未显式设置 seed 时，
        # 各进程的初始随机种子可能不同，因此以 rank 0 的 seed 为准。
        seed = 0
        if shuffle:
            seed = loader.generator.initial_seed() if loader.generator is not None else torch.initial_seed()
            seed = self.broadcast_object(seed)
        if training:
            sampler = DistributedSampler(loader.dataset, num_replicas=self.world_size,
                                         rank=self.rank, shuffle=shuffle, seed=seed,
                                         drop_last=loader.drop_last)
        else:
            sample_count = None
            if loader.drop_last:
                sample_count = len(loader.dataset) // loader.batch_size * loader.batch_size
            sampler = _DistributedEvalSampler(loader.dataset, self.rank, self.world_size,
                                               shuffle=shuffle, seed=seed, sample_count=sample_count)

        # 只更改采样器与需要拆分的批量，保留数据整理、worker 和随机数配置。
        kwargs = {name: getattr(loader, name) for name in (
            'num_workers', 'collate_fn', 'pin_memory', 'drop_last', 'timeout',
            'worker_init_fn', 'multiprocessing_context', 'generator',
            'prefetch_factor', 'persistent_workers', 'pin_memory_device')}
        if hasattr(loader, 'in_order'):
            kwargs['in_order'] = loader.in_order
        if not training:
            kwargs['drop_last'] = False
        return DataLoader(loader.dataset, batch_size=batch_size, sampler=sampler, **kwargs)

    def set_epoch(self, loader, epoch):
        """也允许调用方的自定义 batch_sampler 自行处理逐轮随机采样。"""
        sampler = getattr(loader, 'batch_sampler', None)
        if not hasattr(sampler, 'set_epoch'):
            sampler = getattr(loader, 'sampler', None)
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

    def unwrap_model(self, model):
        """验证仅去掉 DDP，保留 torch.compile 的执行路径。"""
        while isinstance(model, DistributedDataParallel):
            model = model.module
        return model

    def unwrap_serialization_model(self, model):
        """存取权重时去掉 DDP 和编译包装，保持普通模型的 state_dict 键名。"""
        from torch._dynamo.eval_frame import OptimizedModule

        while isinstance(model, (DistributedDataParallel, OptimizedModule)):
            model = model.module if isinstance(model, DistributedDataParallel) else model._orig_mod
        return model

    def sync_model(self, model):
        """初始化回调完成后，以 rank 0 的参数和 buffer 统一各进程模型。"""
        if self.is_distributed:
            with torch.no_grad():
                for parameter in self.unwrap_model(model).parameters():
                    dist.broadcast(parameter, src=0)
                self.sync_buffers(model)

    def sync_buffers(self, model):
        """验证前统一运行统计量，随后允许各进程独立运行不同数量的前向。"""
        if self.is_distributed:
            for buffer in self.unwrap_model(model).buffers():
                dist.broadcast(buffer, src=0)

    def all_gather_object(self, data):
        """收集 CPU 状态；不传递含局部指标函数的 Patch 对象。"""
        if not self.is_distributed:
            return [data]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, data)
        return gathered

    def broadcast_object(self, data, src=0):
        if not self.is_distributed:
            return data
        values = [data if self.rank == src else None]
        dist.broadcast_object_list(values, src=src)
        return values[0]

    @contextmanager
    def ddp_data(self):
        """主进程先完成共享数据准备，其他进程随后执行相同代码；单进程直接执行。"""
        if not self.is_main_process:
            self.barrier()
        try:
            yield
        finally:
            if self.is_main_process:
                self.barrier()

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def close(self):
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False
