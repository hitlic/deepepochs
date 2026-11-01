"""同一脚本通过 python 或 torchrun 完成训练、评估和 Checkpoint 联调。"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import CheckCallback, Trainer, __version__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', default='fp32')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # 每个进程构造相同的数据和模型，随后由 Trainer 负责分片与同步。
    torch.manual_seed(42)
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10 - 1
    targets = (inputs.sum(1) > 0).long()
    train_dl = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=True)
    val_dl = DataLoader(TensorDataset(inputs[:3], targets[:3]), batch_size=4)
    model = nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=.05)

    def accuracy(preds, targets):
        return (preds.argmax(1) == targets).float().mean()

    trainer = Trainer(model, nn.functional.cross_entropy, opt, epochs=3,
                      device=args.device, precision=args.precision, split_batches=True,
                      compile_model=args.compile_model,
                      metrics=[accuracy], callbacks=[CheckCallback('loss', ckpt_dir=args.output)],
                      show_info=False, log_batch=False, running_id='smoke')
    try:
        with trainer.main_process_first():
            marker = Path(args.output) / 'data-ready'
            if trainer.main_process:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text('ready')
            else:
                assert marker.read_text() == 'ready'
        history = trainer.fit(train_dl, val_dl)
        result = trainer.test(val_dl)
        assert len(history['train']['loss']) == 3
        assert math.isfinite(float(result['loss']))
        assert 0 <= float(result['accuracy']) <= 1
        assert set(trainer.model.state_dict()) == {'weight', 'bias'}
        trainer.print(json.dumps({'version': __version__, 'processes': trainer.world_size,
                                 'precision': args.precision, 'compile_model': args.compile_model,
                                 'epochs': len(history['train']['loss']),
                                 'loss': float(result['loss']), 'accuracy': float(result['accuracy'])}))
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
