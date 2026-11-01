"""Measure Trainer throughput with a fixed per-GPU batch size (weak scaling).

Run the same command with one, two and four processes, changing only the expected
world size and output path. The primary timing window includes CPU collation, 
host-to-device transfer, Trainer batch statistics and parameter updates. Epoch
aggregation and epoch logging are outside that window; fit wall time includes them.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import deepepochs
from deepepochs import Callback, Trainer
from deepepochs.distributed import DistributedContext

IMAGE_SHAPE = (3, 224, 224)
CHANNELS = (64, 128, 256, 512)
NUM_CLASSES = 1000
CPU_IMAGE_POOL = 128
SEED = 42


class SyntheticImages(Dataset):
    """A bounded CPU image pool with enough indices for the requested steps."""
    def __init__(self, length):
        self.length = length
        generator = torch.Generator().manual_seed(SEED)
        self.images = torch.randn(CPU_IMAGE_POOL, *IMAGE_SHAPE, generator=generator)
        self.targets = torch.randint(NUM_CLASSES, (CPU_IMAGE_POOL,), generator=generator)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index %= CPU_IMAGE_POOL
        return self.images[index], self.targets[index]


def build_model():
    # Pool early to keep batch=32 safe on a 10 GB card, then spend compute on
    # eight convolution layers with a modest gradient communication volume.
    layers = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
              nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)]
    in_channels = 64
    for stage, out_channels in enumerate(CHANNELS):
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=1 if stage == 0 else 2,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ])
        in_channels = out_channels
    layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, NUM_CLASSES)])
    return nn.Sequential(*layers)


class BatchTiming(Callback):
    def __init__(self, warmup, steps):
        # Run first before a batch and last after it, around all batch callbacks.
        super().__init__(priority=-1)
        self.warmup = warmup
        self.steps = steps
        self.batch_seconds = []
        self.samples = 0
        self.window_start = None
        self.window_end = None
        self.batch_start = None

    def start_window(self, trainer):
        torch.cuda.synchronize(trainer.device)
        trainer.distributed.barrier()
        torch.cuda.synchronize(trainer.device)
        torch.cuda.reset_peak_memory_stats(trainer.device)
        self.window_start = time.perf_counter()

    def on_before_train_epoch(self, trainer, task):
        if self.warmup == 0:
            self.start_window(trainer)

    def on_before_train_batch(self, trainer, batch_x, batch_y, batch_idx):
        if batch_idx >= self.warmup:
            # Trainer has already copied this batch to the GPU. This secondary
            # timer isolates the batch work; the outer window also includes H2D.
            torch.cuda.synchronize(trainer.device)
            self.batch_start = time.perf_counter()

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        if batch_idx < self.warmup:
            if batch_idx + 1 == self.warmup:
                self.start_window(trainer)
            return
        torch.cuda.synchronize(trainer.device)
        end = time.perf_counter()
        self.batch_seconds.append(end - self.batch_start)
        self.samples += trainer.current_task.current_batch_size
        if len(self.batch_seconds) == self.steps:
            self.window_end = end

    def rank_result(self, trainer, fit_seconds):
        if len(self.batch_seconds) != self.steps or self.window_end is None:
            raise RuntimeError('The Trainer did not execute the requested measured steps.')
        elapsed = self.window_end - self.window_start
        ordered = sorted(self.batch_seconds)
        return {
            'rank': trainer.rank,
            'local_rank': trainer.distributed.local_rank,
            'device': str(trainer.device),
            'device_name': torch.cuda.get_device_name(trainer.device),
            'device_total_memory_bytes': torch.cuda.get_device_properties(trainer.device).total_memory,
            'measured_samples': self.samples,
            'measured_window_seconds': elapsed,
            'samples_per_second': self.samples / elapsed,
            'batch_seconds_mean': statistics.mean(self.batch_seconds),
            'batch_seconds_median': statistics.median(self.batch_seconds),
            'batch_seconds_p95': ordered[math.ceil(0.95 * len(ordered)) - 1],
            'batch_seconds_max': max(self.batch_seconds),
            'batch_seconds': self.batch_seconds,
            'fit_wall_seconds': fit_seconds,
            'grad_scaler_scale': float(trainer.scaler.get_scale()),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(trainer.device),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(trainer.device),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='JSON file written by rank 0')
    parser.add_argument('--expected-world-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200, help='Measured batches per rank')
    parser.add_argument('--warmup', type=int, default=20, help='Unmeasured batches per rank')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.batch_size < 1 or args.expected_world_size < 1:
        parser.error('steps, batch-size and expected-world-size must be positive; warmup must be nonnegative')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This throughput benchmark requires CUDA.')
    # Initialize once before sizing the dataset. Trainer borrows this process
    # group, and the outer context remains responsible for its cleanup.
    context = DistributedContext('cuda')
    trainer = None
    try:
        if context.world_size != args.expected_world_size:
            raise RuntimeError(f'Expected {args.expected_world_size} processes, got {context.world_size}.')
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('The selected GPU does not support BF16.')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        total_steps = args.warmup + args.steps
        dataset_length = total_steps * args.batch_size * context.world_size
        dataset = SyntheticImages(dataset_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=0, pin_memory=True)
        model = build_model()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        timing = BatchTiming(args.warmup, args.steps)
        trainer = Trainer(model, nn.functional.cross_entropy, optimizer, epochs=1,
                          device=context.device, precision=args.precision,
                          split_batches=False, callbacks=[timing], metrics=[],
                          show_info=False, log_batch=False, log_tqdm=False)
        torch.cuda.synchronize(context.device)
        context.barrier()
        torch.cuda.synchronize(context.device)
        start = time.perf_counter()
        history = trainer.fit(loader)
        torch.cuda.synchronize(context.device)
        fit_seconds = time.perf_counter() - start
        loss = float(history['train']['loss'][-1])
        if not math.isfinite(loss):
            raise RuntimeError('The benchmark loss became non-finite.')
        per_rank = context.all_gather_object(timing.rank_result(trainer, fit_seconds))
        slowest_seconds = max(item['measured_window_seconds'] for item in per_rank)
        fit_wall_seconds = max(item['fit_wall_seconds'] for item in per_rank)
        measured_samples = sum(item['measured_samples'] for item in per_rank)
        report = {
            'deepepochs_version': deepepochs.__version__,
            'deepepochs_file': deepepochs.__file__,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'world_size': context.world_size,
            'distributed_backend': torch.distributed.get_backend() if context.is_distributed else None,
            'scaling_mode': 'weak; fixed batch size per GPU',
            'precision': args.precision,
            'seed': SEED,
            'tf32': False,
            'cudnn_benchmark': True,
            'cudnn_deterministic': False,
            'cpu_threads_per_rank': 1,
            'warmup_steps': args.warmup,
            'measured_steps': args.steps,
            'batch_size_per_rank': args.batch_size,
            'global_batch_size': args.batch_size * context.world_size,
            'gradient_accumulation_steps': 1,
            'split_batches': False,
            'input_shape_per_rank': [args.batch_size, *IMAGE_SHAPE],
            'model': {'name': 'small_conv9', 'stage_channels': list(CHANNELS),
                      'classes': NUM_CLASSES, 'parameters': parameter_count, 'layout': 'nchw'},
            'optimizer': {'name': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
            'data': {'source': 'cached synthetic CPU images', 'cached_images_per_rank': CPU_IMAGE_POOL,
                     'dataset_length': dataset_length, 'num_workers': 0, 'pin_memory': True},
            'timing_scope': {
                'measured_window': 'after warmup through final batch; includes CPU collation, H2D, Trainer updates and batch statistics',
                'batch_callback': 'before_train_batch after H2D through after_train_batch; CUDA synchronize at both ends',
                'fit_wall': 'entire Trainer.fit including warmup, initialization callbacks, epoch aggregation and logging',
            },
            'measured_global_samples': measured_samples,
            'slowest_rank_window_seconds': slowest_seconds,
            'global_samples_per_second': measured_samples / slowest_seconds,
            'fit_wall_seconds': fit_wall_seconds,
            'fit_global_samples_per_second': dataset_length / fit_wall_seconds,
            'final_epoch_loss': loss,
            'per_rank': per_rank,
        }
        if context.is_main_process:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(report))
    finally:
        if trainer is not None:
            trainer.close()
        context.close()


if __name__ == '__main__':
    main()
