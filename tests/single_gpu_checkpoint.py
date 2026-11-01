"""普通 Python 进程恢复 multigpu_evaluation 的产物，并继续一轮训练。 """

import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import torch
from multigpu_evaluation import FixedLogits, classification_data, quiet
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, CheckCallback, EpochTask, Optimizer, Trainer, metrics


class AdvanceMarker(Callback):
    def on_after_epoch(self, trainer, **kwargs):
        with torch.no_grad():
            trainer.model.anchor.add_(1.)


def train_f1(preds, labels):
    return metrics.f1(preds, labels, average='macro')


def val_score(preds, labels):
    return (preds.argmax(1) == labels).float().mean()


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def test_marker(trainer, model, loader, expected_marker):
    current_marker = model.anchor.item()
    current_epoch = trainer.init_epoch
    optimizer_state = deepcopy(trainer.opt.state_dict())
    scaler_state = deepcopy(trainer.scaler.state_dict())
    seen = []

    def observe(preds, labels):
        seen.append(model.anchor.item())
        return preds.mean()

    quiet(trainer.test, loader, metrics=[observe])
    assert seen and all(value == expected_marker for value in seen), seen
    assert model.anchor.item() == current_marker
    assert trainer.init_epoch == current_epoch
    # 此用例使用无动量 SGD，状态只包含标量和参数编号，可直接比较。
    assert trainer.opt.state_dict() == optimizer_state
    assert trainer.scaler.state_dict() == scaler_state
    return seen[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-world-size', type=int, choices=(2, 4), default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    source = args.checkpoint.resolve()
    source_state = read(source)
    assert source_state['epoch'] == 2 and source_state['model_state']['anchor'].item() == 3.
    before_move = args.output / 'before-move'
    relocated = args.output / 'relocated'
    shutil.copytree(source.parent, before_move)
    shutil.move(str(before_move), relocated)
    assert not before_move.exists() and source.is_file()
    checkpoint = relocated / source.name

    first = CheckCallback('loss', on_stage='val', patience=2, save_best=True,
                           ckpt_dir=str(args.output / 'continued'))
    second = CheckCallback('val_score', on_stage='val', mode='max', save_best=False,
                            ckpt_dir=str(args.output / 'continued'))
    model = FixedLogits()
    raw = torch.optim.SGD(model.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(raw, patience=0, factor=0.5)
    trainer = Trainer(model, F.cross_entropy, Optimizer(raw, scheduler, sched_with_loss=True),
                      device=args.device, precision=args.precision, epochs=4,
                      callbacks=[first, second, AdvanceMarker()], running_id='single-gpu',
                      show_info=False, log_batch=False)
    records = []

    def record(case, **details):
        item = {'case': case, 'status': 'PASS', 'precision': args.precision, **details}
        records.append(item)
        print(json.dumps(item), flush=True)

    try:
        assert not trainer.distributed.is_distributed and trainer.world_size == 1
        quiet(trainer.load_checkpoint, str(checkpoint))
        assert trainer.init_epoch == 3 and model.anchor.item() == 3.
        assert raw.param_groups[0]['lr'] == 0.02
        assert scheduler.state_dict() == source_state['opt_state'][0]['sched_state']
        restored_scaler = deepcopy(trainer.scaler.state_dict())
        if trainer.scaler.is_enabled():
            assert restored_scaler == source_state['scaler_state']
            assert restored_scaler['growth_interval'] == 2000
            assert restored_scaler['_growth_tracker'] == 6
        assert first.worse_times == second.worse_times == 2
        assert Path(first.ckpt_path).parent == relocated
        assert Path(second.ckpt_path) == checkpoint
        assert Path(first.ckpt_path) != Path(second.ckpt_path)
        assert [read(callback.ckpt_path)['model_state']['anchor'].item()
                for callback in (first, second)] == [1., 3.]
        record('distributed_checkpoint_restored_in_one_process',
               source_world_size=args.source_world_size, world_size=trainer.world_size,
               device=str(trainer.device), epoch=trainer.init_epoch,
               learning_rate=raw.param_groups[0]['lr'], marker=model.anchor.item(),
               scaler_state=restored_scaler)

        train_logits, train_targets = classification_data(args.source_world_size * 2)
        val_logits, val_targets = classification_data(args.source_world_size * 2 + 1)
        # 原来每卡批量为 2；单卡使用完整全局批量，保留每个 Task 的更新次数。
        train_loader = DataLoader(TensorDataset(train_logits, train_targets),
                                  batch_size=args.source_world_size * 2)
        val_loader = DataLoader(TensorDataset(val_logits, val_targets), batch_size=2)
        tiny_loader = DataLoader(TensorDataset(val_logits[:1], val_targets[:1]), batch_size=2)
        observed = test_marker(trainer, model, val_loader, expected_marker=1.)
        record('relocated_monitor_ownership', selected_marker=observed,
               current_marker=model.anchor.item(),
               monitor_paths=[first.ckpt_path, second.ckpt_path])

        optimizer_steps = []
        handle = raw.register_step_post_hook(lambda *unused: optimizer_steps.append(True))
        try:
            history = quiet(trainer.fit,
                            train_tasks=[EpochTask(train_loader, metrics=[train_f1]), EpochTask(train_loader)],
                            val_tasks=[EpochTask(tiny_loader), EpochTask(val_loader, metrics=[val_score])])
        finally:
            handle.remove()
        assert len(history['train']['loss']) == 1 and trainer.init_epoch == 4
        assert len(optimizer_steps) == 2
        assert model.anchor.item() == 4.
        assert raw.param_groups[0]['lr'] == 0.01
        continued = read(second.ckpt_path)
        assert continued['epoch'] == 3 and continued['model_state']['anchor'].item() == 4.
        assert Path(second.ckpt_path).parent == args.output / 'continued' / 'single-gpu'
        continued_scaler = deepcopy(trainer.scaler.state_dict())
        if trainer.scaler.is_enabled():
            # 本用例原来有效更新 6 次，续训再更新 2 次，尚未达到增长周期。
            expected_scaler = deepcopy(restored_scaler)
            expected_scaler['_growth_tracker'] = 8
            assert continued_scaler == expected_scaler
            assert continued['scaler_state'] == continued_scaler
        record('one_gpu_training_continued', epoch=trainer.init_epoch,
               optimizer_steps=len(optimizer_steps), global_batch_size=args.source_world_size * 2,
               learning_rate=raw.param_groups[0]['lr'], marker=model.anchor.item(),
               checkpoint=second.ckpt_path, scaler_state=continued_scaler)

        expected_marker = read(first.ckpt_path)['model_state']['anchor'].item()
        observed = test_marker(trainer, model, val_loader, expected_marker)
        record('testing_preserves_continued_training_state',
               selected_checkpoint=first.ckpt_path, selected_marker=observed,
               restored_marker=model.anchor.item(), epoch=trainer.init_epoch,
               learning_rate=raw.param_groups[0]['lr'], scaler_state=trainer.scaler.state_dict())
        report = {'source_checkpoint': str(source), 'relocated_checkpoint': str(checkpoint),
                  'source_world_size': args.source_world_size, 'world_size': 1,
                  'precision': args.precision, 'restored_scaler_state': restored_scaler,
                  'continued_scaler_state': continued_scaler,
                  'fixture': 'multigpu_evaluation.FixedLogits', 'cases': records}
        (args.output / 'single-gpu-checkpoint.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
