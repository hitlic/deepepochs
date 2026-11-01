"""通过launch完成单进程或单机多GPU训练、评估和日志收尾。 """
import argparse
import contextlib
import io
import json
import math
import os
import signal
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, LogCallback, Trainer, __version__, launch


class OutputDtypes(Callback):
    def __init__(self):
        super().__init__()
        self.values = set()

    def on_after_train_forward(self, trainer, model_out):
        self.values.add(str(model_out.dtype))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=None)
    parser.add_argument('--expected-world-size', type=int)
    parser.add_argument('--batch-size', type=int, default=4, help='各进程平分的全局批量，需能被进程数整除')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def train():
    args = arguments()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10 - 1
    targets = (inputs.sum(1) > 0).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=args.batch_size)
    validation = DataLoader(TensorDataset(inputs[:3], targets[:3]), batch_size=args.batch_size)
    model = nn.Linear(3, 2)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    check = CheckCallback('loss', ckpt_dir=str(args.output / 'checkpoints'))
    logger = LogCallback(log_dir=str(args.output / 'logs'))
    dtypes = OutputDtypes()

    def accuracy(preds, labels):
        return (preds.argmax(1) == labels).float().mean()

    trainer = Trainer(model, nn.functional.cross_entropy, optimizer, epochs=3,
                      device=args.device, precision=args.precision, split_batches=True,
                      callbacks=[check, logger, dtypes], metrics=[accuracy],
                      running_id='launch', show_info=False, log_batch=False)
    writer_thread = None
    try:
        if args.expected_world_size is not None:
            assert trainer.world_size == args.expected_world_size
        assert trainer.rank == int(os.environ.get('RANK', '0'))
        assert trainer.distributed.local_rank == int(os.environ.get('LOCAL_RANK', '0'))
        if args.device == 'cuda':
            assert trainer.device.index == trainer.distributed.local_rank
            assert torch.cuda.current_device() == trainer.device.index
        with contextlib.redirect_stdout(io.StringIO()):
            history = trainer.fit(loader, validation)
            epoch_before_test = trainer.init_epoch
            scaler_before_test = trainer.scaler.state_dict()
            results = trainer.test(validation)
        assert len(history['train']['loss']) == 3 and trainer.init_epoch == epoch_before_test == 3
        assert trainer.scaler.state_dict() == scaler_before_test
        assert math.isfinite(float(results['loss']))
        assert any(not torch.equal(value.detach().cpu(), initial[name])
                   for name, value in model.state_dict().items())
        expected_dtype = {'fp32': 'torch.float32', 'fp16': 'torch.float16', 'bf16': 'torch.bfloat16'}[args.precision]
        assert dtypes.values == {expected_dtype}, dtypes.values
        checkpoint = torch.load(check.ckpt_path, map_location='cpu', weights_only=False)
        reference = nn.Linear(3, 2).to(trainer.device)
        reference.load_state_dict(checkpoint['model_state'])
        with torch.no_grad(), trainer.autocast():
            predictions = reference(inputs[:3].to(trainer.device))
            expected_loss = nn.functional.cross_entropy(predictions, targets[:3].to(trainer.device))
            expected_accuracy = accuracy(predictions, targets[:3].to(trainer.device))
        tolerance = {'fp32': 1e-6, 'fp16': 2e-3, 'bf16': 1e-2}[args.precision]
        assert abs(float(results['loss']) - float(expected_loss)) < tolerance
        assert abs(float(results['accuracy']) - float(expected_accuracy)) < 1e-6
        if trainer.main_process:
            writer_thread = trainer.logger.file_writer.event_writer._async_writer._worker
            assert writer_thread.is_alive()
        state = {'rank': trainer.rank, 'local_rank': trainer.distributed.local_rank,
                 'pid': os.getpid(), 'device': str(trainer.device),
                 'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
                 'loss': float(results['loss']), 'accuracy': float(results['accuracy']),
                 'scaler': trainer.scaler.state_dict(),
                 'parameters': {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}}
        ranks = trainer.distributed.all_gather_object(state)
        assert [item['rank'] for item in ranks] == list(range(trainer.world_size))
        assert all(item['parameters'] == ranks[0]['parameters'] and item['scaler'] == ranks[0]['scaler']
                   and item['loss'] == ranks[0]['loss'] and item['accuracy'] == ranks[0]['accuracy']
                   for item in ranks)
        result = {'status': 'PASS', 'version': __version__, 'rank': trainer.rank,
                  'world_size': trainer.world_size, 'precision': args.precision, 'epochs': trainer.init_epoch,
                  'global_batch_size': args.batch_size,
                  'loss': state['loss'], 'accuracy': state['accuracy'], 'per_rank': ranks,
                  'checkpoint': check.ckpt_path, 'global_metric_reference': True}
    finally:
        trainer.close()
        assert writer_thread is None or not writer_thread.is_alive()
    assert not dist.is_initialized()
    result['writer_closed'] = True
    result['process_group_closed'] = True
    return result


def main():
    args = arguments()
    before = os.environ.get('OMP_NUM_THREADS')
    signals = {getattr(signal, name): signal.getsignal(getattr(signal, name))
               for name in ('SIGINT', 'SIGTERM', 'SIGHUP', 'SIGQUIT') if hasattr(signal, name)}
    parent_pid = os.getpid()
    result = launch(train, num_processes=args.num_processes)
    assert os.environ.get('OMP_NUM_THREADS') == before
    assert all(signal.getsignal(number) == handler for number, handler in signals.items())
    if result['rank'] == 0:
        result['launcher_parent_pid'] = parent_pid
        result['parent_omp_before'] = before
        result['parent_omp_after'] = os.environ.get('OMP_NUM_THREADS')
        result['signals_restored'] = True
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / 'launch.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
