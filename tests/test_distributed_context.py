"""直接比较分片、全局梯度和进程组归属；不依赖下载数据或 GPU。"""

import os
import socket
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Sampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from deepepochs.distributed import DistributedContext


class ReverseSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset) - 1, -1, -1))

    def __len__(self):
        return len(self.dataset)


def _collate_values(items):
    return torch.stack([item[0] for item in items])


def _distributed_checks(rank, port, directory):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='2',
                      RANK=str(rank), LOCAL_RANK=str(rank))
    torch.manual_seed(3 + rank)
    context = DistributedContext('cpu')
    try:
        assert context.rank == rank and context.local_rank == rank
        assert context.world_size == 2 and context.is_distributed
        assert context.is_main_process == (rank == 0)
        assert context.device == torch.device('cpu')

        # 验证初次下载对应的顺序：非主进程进入块时，主进程的缓存已存在。
        marker = Path(directory) / 'prepared'
        with context.main_process_first():
            if context.is_main_process:
                marker.write_text('ready')
            else:
                assert marker.read_text() == 'ready'
        context.barrier()

        dataset = TensorDataset(torch.arange(7.).view(-1, 1))
        generator = torch.Generator().manual_seed(101 + rank)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator,
                            collate_fn=_collate_values)
        train_loader = context.prepare_dataloader(loader)
        assert train_loader.batch_size == 2
        assert train_loader.generator is generator
        assert train_loader.collate_fn is _collate_values
        assert len(train_loader) == 2
        context.set_epoch(train_loader, 4)
        expected = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=101)
        expected.set_epoch(4)
        assert list(train_loader.sampler) == list(expected)
        # 两个进程使用同一排列后再取分片；只有训练所需的一个补齐样本。
        train_indices = context.all_gather_object(list(train_loader.sampler))
        assert len(train_indices[0] + train_indices[1]) == 8
        assert set(train_indices[0] + train_indices[1]) == set(range(7))

        split_loader = context.prepare_dataloader(DataLoader(dataset, batch_size=4), split_batches=True)
        assert split_loader.batch_size == 2
        try:
            context.prepare_dataloader(DataLoader(dataset, batch_size=3), split_batches=True)
        except ValueError:
            pass
        else:
            raise AssertionError('不可整除的批量应被拒绝。')

        prepared = DataLoader(dataset, batch_size=2,
                              sampler=DistributedSampler(dataset, num_replicas=2, rank=rank))
        assert context.prepare_dataloader(prepared) is prepared
        custom = DataLoader(dataset, batch_size=2, sampler=ReverseSampler(dataset))
        assert context.prepare_dataloader(custom) is custom
        custom_batch = DataLoader(dataset, batch_sampler=BatchSampler(ReverseSampler(dataset), 2, False))
        assert context.prepare_dataloader(custom_batch) is custom_batch

        # drop_last=True 不向训练分片补齐，并保持本地只产出完整批量。
        dropped = context.prepare_dataloader(DataLoader(dataset, batch_size=2, drop_last=True))
        assert dropped.drop_last and dropped.sampler.drop_last
        assert all(len(batch[0]) == 2 for batch in dropped)

        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        model.register_buffer('running_count', torch.tensor(float(rank)))
        wrapped = context.wrap_model(model)
        assert context.unwrap_model(wrapped) is model
        model.running_count.fill_(rank + 1.)
        context.sync_buffers(wrapped)
        assert model.running_count.item() == 1.

        with torch.no_grad():
            model.weight.fill_(rank + 0.25)
            model.running_count.fill_(rank + 2.)
        context.sync_model(wrapped)
        torch.testing.assert_close(model.weight, torch.tensor([[0.25]]))
        assert model.running_count.item() == 2.
        model.weight.data.zero_()

        # 各进程分别使用 x=1 和 x=2；全局 MSE 的 SGD 更新应为 weight=1。
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        x = torch.tensor([[float(rank + 1)]])
        nn.functional.mse_loss(wrapped(x), 2 * x).backward()
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.tensor([[1.]]))

        # 验证分片长度分别为 3 和 2，批量数分别为 2 和 1。原始模型的
        # 前向没有逐批通信，只有任务末尾的收集，因此两边可以正常会合。
        validation = TensorDataset(torch.arange(5.).view(-1, 1))
        eval_loader = context.prepare_dataloader(DataLoader(validation, batch_size=2), training=False)
        assert len(eval_loader.sampler) == (3 if rank == 0 else 2)
        assert len(eval_loader) == (2 if rank == 0 else 1)
        context.sync_buffers(wrapped)
        with torch.no_grad():
            predictions = [context.unwrap_model(wrapped)(batch[0]).flatten().tolist() for batch in eval_loader]
        gathered = context.all_gather_object([value for batch in predictions for value in batch])
        assert sorted(gathered[0] + gathered[1]) == [0., 1., 2., 3., 4.]

        tiny = context.prepare_dataloader(DataLoader(TensorDataset(torch.tensor([1.])), batch_size=2),
                                           training=False)
        assert len(tiny) == (1 if rank == 0 else 0)
        tiny_indices = context.all_gather_object(list(tiny.sampler))
        assert tiny_indices == [[0], []]

        # 原始 loader 会评估 6 个样本中的前 4 个；先分片再丢尾会把它们全丢掉。
        six = TensorDataset(torch.arange(6.).view(-1, 1))
        for split_batches in (False, True):
            whole_batches = context.prepare_dataloader(
                DataLoader(six, batch_size=4, drop_last=True),
                training=False, split_batches=split_batches)
            assert not whole_batches.drop_last
            selected = [value for batch in whole_batches for value in batch[0].flatten().tolist()]
            selected = context.all_gather_object(selected)
            assert sorted(selected[0] + selected[1]) == [0., 1., 2., 3.], selected

        generator = torch.Generator().manual_seed(17 + rank)
        shuffled = context.prepare_dataloader(
            DataLoader(six, batch_size=4, shuffle=True, generator=generator, drop_last=True),
            training=False)
        selected = context.all_gather_object(list(shuffled.sampler))
        expected = torch.randperm(6, generator=torch.Generator().manual_seed(17)).tolist()[:4]
        assert sorted(selected[0] + selected[1]) == sorted(expected), selected

        state = context.broadcast_object({'epoch': 3, 'value': 2.} if rank == 0 else None)
        assert state == {'epoch': 3, 'value': 2.}
        states = context.all_gather_object(torch.tensor(rank))
        assert [value.item() for value in states] == [0, 1]
        assert all(value.device.type == 'cpu' for value in states)

        # 第二个 context 仅使用已有进程组；关闭它不能影响创建者。
        borrowed = DistributedContext('cpu')
        borrowed.close()
        assert dist.is_initialized()
        context.barrier()
    finally:
        context.close()
    assert not dist.is_initialized()


class DistributedContextTests(unittest.TestCase):
    def test_single_process_preserves_loader_and_model(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
            context = DistributedContext('cpu')
        loader = DataLoader(TensorDataset(torch.arange(3)), batch_size=2)
        self.assertIs(context.prepare_dataloader(loader, split_batches=True), loader)
        model = nn.Linear(1, 1)
        self.assertIs(context.wrap_model(model), model)
        self.assertIs(context.unwrap_model(model), model)
        self.assertEqual(context.all_gather_object({'value': 3}), [{'value': 3}])
        self.assertEqual(context.broadcast_object('state'), 'state')
        self.assertFalse(context.is_distributed)
        self.assertTrue(context.is_main_process)
        context.sync_buffers(model)
        context.sync_model(model)
        context.barrier()
        context.close()

    def test_serialization_unwrap_preserves_ordinary_orig_mod_fields(self):
        class ModelWithOrigMod(nn.Module):
            def __init__(self):
                super().__init__()
                self._orig_mod = nn.Linear(1, 1)

            def forward(self, value):
                return self._orig_mod(value)

        context = DistributedContext('cpu')
        model = ModelWithOrigMod()
        compiled = torch.compile(model, backend='eager')
        self.assertIs(context.unwrap_model(compiled), compiled)
        self.assertIs(context.unwrap_serialization_model(compiled), model)
        self.assertIs(context.unwrap_serialization_model(model), model)
        self.assertEqual(set(context.unwrap_serialization_model(compiled).state_dict()),
                         {'_orig_mod.weight', '_orig_mod.bias'})
        context.close()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要 Gloo')
    def test_two_cpu_processes(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_checks, args=(port, directory), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
