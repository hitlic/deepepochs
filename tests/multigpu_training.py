"""torchrun 下比较 CUDA/NCCL 训练与独立的完整全局批量参考。 """

import argparse
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Callback, EpochTask, GradClipCallback, Optimizer, Trainer
from deepepochs.distributed import DistributedContext
from deepepochs.loops import StopLoopException


def linear_model(device):
    model = nn.Linear(3, 1).to(device)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.125, -0.25, 0.375]], device=device))
        model.bias.fill_(0.0625)
    model.register_buffer('counter', torch.zeros(1, device=device))
    return model


def regression_data(size):
    inputs = (torch.arange(size * 3, dtype=torch.float32).reshape(size, 3) % 17 - 8) / 32
    targets = inputs.sum(1, keepdim=True) * 0.25 + 0.03125
    return inputs, targets


def parameters(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def gradients(model):
    return {name: value.grad.detach().cpu().clone()
            for name, value in model.named_parameters() if value.grad is not None}


class Probe(Callback):
    def __init__(self):
        # 在 GradClipCallback 后读取真正用于更新的反缩放梯度。
        super().__init__(priority=2)
        self.gradients = []
        self.dtypes = set()

    def on_before_optimize(self, trainer, optimize_index):
        model = trainer.distributed.unwrap_serialization_model(trainer.model.model)
        self.gradients.append(gradients(model))

    def on_after_train_forward(self, trainer, model_out):
        if isinstance(model_out, torch.Tensor):
            self.dtypes.add(model_out.dtype)


def make_trainer(model, optimizer, precision, **kwargs):
    trainer = Trainer(model, F.mse_loss, optimizer, device='cuda', precision=precision,
                      epochs=1, show_info=False, log_batch=False, **kwargs)
    if trainer.scaler.is_enabled():
        # 固定小尺度避免普通测试因初始 scale 偶发溢出；两次有效更新会增长一次。
        trainer.scaler.load_state_dict({'scale': 128., 'growth_factor': 2.,
                                        'backoff_factor': 0.5, 'growth_interval': 2,
                                        '_growth_tracker': 0})
    return trainer


def fit(trainer, loader=None, **kwargs):
    with redirect_stdout(io.StringIO()):
        return trainer.fit(loader, **kwargs)


def step_counters(optimizers):
    counts = [0] * len(optimizers)
    handles = []
    for index, optimizer in enumerate(optimizers):
        def count_step(*args, index=index):
            counts[index] += 1
        handles.append(optimizer.register_step_post_hook(count_step))
    return counts, handles


def assert_close(actual, expected, precision, gradient=False):
    if precision == 'fp32':
        atol, rtol = 2e-7, 2e-5
    elif precision == 'fp16':
        atol, rtol = (3e-4, 5e-3) if gradient else (5e-5, 1e-3)
    else:
        atol, rtol = (3e-3, 3e-2) if gradient else (4e-4, 5e-3)
    assert set(actual) == set(expected), (set(actual), set(expected))
    maximum = 0.
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, atol=atol, rtol=rtol, msg=lambda message: f'{name}: {message}')
        maximum = max(maximum, float((actual[name] - value).abs().max()))
    return maximum


def assert_rank_state(context, model, trainer, counts, schedulers=()):
    # 同步参数应逐位一致；与独立 FP32 参考的精度容差在另一处单独比较。
    state = {'parameters': parameters(model), 'scaler': trainer.scaler.state_dict(),
             'steps': list(counts),
             'schedulers': [scheduler.state_dict() for scheduler in schedulers]}
    gathered = context.all_gather_object(state)
    for other in gathered:
        assert_close(other['parameters'], gathered[0]['parameters'], 'fp32')
        for name in state['parameters']:
            torch.testing.assert_close(other['parameters'][name], gathered[0]['parameters'][name], rtol=0, atol=0)
        assert other['scaler'] == gathered[0]['scaler'], gathered
        assert other['steps'] == gathered[0]['steps'], gathered
        assert other['schedulers'] == gathered[0]['schedulers'], gathered


def device_collectives(context):
    assert dist.get_backend() == 'nccl', dist.get_backend()
    assert torch.cuda.current_device() == context.local_rank == context.device.index
    devices = context.all_gather_object({'rank': context.rank, 'local_rank': context.local_rank,
                                         'device': context.device.index,
                                         'name': torch.cuda.get_device_name(context.device)})
    assert sorted(item['device'] for item in devices) == list(range(context.world_size)), devices
    value = torch.tensor(float(context.rank + 1), device=context.device)
    dist.all_reduce(value)
    expected = context.world_size * (context.world_size + 1) / 2
    assert value.item() == expected, value
    return {'backend': dist.get_backend(), 'devices': devices, 'all_reduce_sum': value.item()}


def global_batch_case(context, precision, local_samples, accumulation=1, split=False):
    size = context.world_size * local_samples
    inputs, targets = regression_data(size)
    global_batch = context.world_size * 2 * accumulation
    reference = linear_model('cpu')
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected_gradients = []
    for start in range(0, size, global_batch):
        reference_optimizer.zero_grad()
        F.mse_loss(reference(inputs[start:start + global_batch]),
                   targets[start:start + global_batch]).backward()
        expected_gradients.append(gradients(reference))
        reference_optimizer.step()

    model = linear_model(context.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # 学习率保持不变以便单独比较梯度与权重；step scheduler 的跟随由多优化器用例验证。
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)
    probe = Probe()
    counts, handles = step_counters([optimizer])
    trainer = make_trainer(model, Optimizer(optimizer, scheduler, sched_on='step'), precision,
                           callbacks=[probe], split_batches=split,
                           gradient_accumulation_steps=accumulation)
    try:
        loader_batch = context.world_size * 2 if split else 2
        fit(trainer, DataLoader(TensorDataset(inputs, targets), batch_size=loader_batch))
        assert counts == [len(expected_gradients)], counts
        assert len(probe.gradients) == len(expected_gradients)
        gradient_error = max(assert_close(actual, expected, precision, gradient=True)
                             for actual, expected in zip(probe.gradients, expected_gradients))
        weight_error = assert_close(parameters(model), parameters(reference), precision)
        expected_dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
        assert probe.dtypes == {expected_dtype}, probe.dtypes
        assert scheduler.last_epoch == len(expected_gradients), scheduler.state_dict()
        assert_rank_state(context, model, trainer, counts, [scheduler])
        return {'samples': size, 'global_batch': global_batch, 'updates': counts[0],
                'max_gradient_error': gradient_error, 'max_weight_error': weight_error,
                'scaler': trainer.scaler.state_dict()}
    finally:
        for handle in handles:
            handle.remove()
        trainer.close()


def interrupted_accumulation_case(context, precision, event):
    class RecordLoss:
        def __init__(self):
            self.values = []

        def step(self, loss):
            self.values.append(float(loss))

        def state_dict(self):
            return {'values': self.values}

    class Interrupt(Callback):
        def __init__(self):
            super().__init__()
            self.enabled = True

        def on_after_train_batch(self, trainer, metrics, batch_idx):
            if self.enabled and event == 'batch':
                self.enabled = False
                raise StopLoopException('stop after no_sync microbatch')

        def on_before_optimize(self, trainer, optimize_index):
            if self.enabled and event == 'optimize':
                self.enabled = False
                raise RuntimeError('stop before optimizer update')

    # 顺序分布式采样让各rank依次获得标签[1, 3]；完整全局批量的
    # 独立参考均值损失为5，更新后的权重为0.4。
    inputs = torch.ones(context.world_size * 2, 1)
    targets = torch.cat((torch.ones(context.world_size, 1), torch.full((context.world_size, 1), 3.)))
    reference = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(reference.weight)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    F.mse_loss(reference(inputs), targets).backward()
    reference_optimizer.step()

    model = nn.Linear(1, 1, bias=False).to(context.device)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = RecordLoss()
    counts, handles = step_counters([optimizer])
    trainer = make_trainer(model, Optimizer(optimizer, scheduler, sched_on='step', sched_with_loss=True),
                           precision, callbacks=[Interrupt()], gradient_accumulation_steps=2)
    initial_scaler = trainer.scaler.state_dict()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    try:
        if event == 'batch':
            fit(trainer, loader)
        else:
            try:
                fit(trainer, loader)
            except RuntimeError as error:
                assert str(error) == 'stop before optimizer update', error
            else:
                raise AssertionError('interruption did not reach caller')
        assert counts == [0], counts
        assert scheduler.values == [], scheduler.values
        assert model.weight.grad is None
        assert float(model.weight.detach()) == 0.
        assert trainer.scaler.state_dict() == initial_scaler
        fit(trainer, loader)
        assert counts == [1], counts
        assert scheduler.values == [5.], scheduler.values
        weight_error = assert_close(parameters(model), parameters(reference), precision)
        assert_rank_state(context, model, trainer, counts, [scheduler])
        return {'interrupted_at': event, 'updates': counts[0], 'scheduler_loss': scheduler.values,
                'max_weight_error': weight_error, 'scaler': trainer.scaler.state_dict()}
    finally:
        for handle in handles:
            handle.remove()
        trainer.close()


def initialization_case(context, precision):
    class Initialize(Callback):
        def on_before_fit(self, trainer, epochs):
            model = context.unwrap_serialization_model(trainer.model.model)
            generator = torch.Generator().manual_seed(700 + context.rank)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator).to(context.device) / 8)
                model.counter.fill_(context.rank + 1.)
            self.initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        def on_before_train_epochs(self, trainer, tasks, epoch_idx):
            model = context.unwrap_serialization_model(trainer.model.model)
            expected = context.broadcast_object(self.initial if context.is_main_process else None)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
            self.checked = True

    callback = Initialize()
    model = linear_model(context.device)
    trainer = make_trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), precision,
                           callbacks=[callback])
    try:
        inputs, targets = regression_data(context.world_size * 2)
        fit(trainer, DataLoader(TensorDataset(inputs, targets), batch_size=2))
        assert callback.checked
        assert_rank_state(context, model, trainer, [1])
        return {'callback_parameters_synchronized': True, 'buffer': model.counter.item()}
    finally:
        trainer.close()


def shared_optimizers_case(context, precision):
    inputs, targets = regression_data(context.world_size * 2)
    reference = linear_model('cpu')
    reference_optimizers = [torch.optim.SGD(reference.parameters(), lr=rate) for rate in (0.03, 0.07)]
    F.mse_loss(reference(inputs), targets).backward()
    nn.utils.clip_grad_norm_(reference.parameters(), 0.05)
    expected_gradient = gradients(reference)
    for optimizer in reference_optimizers:
        optimizer.step()

    model = linear_model(context.device)
    optimizers = [torch.optim.SGD(model.parameters(), lr=rate) for rate in (0.03, 0.07)]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5) for opt in optimizers]
    wrapped = [Optimizer(opt, scheduler, sched_on='step') for opt, scheduler in zip(optimizers, schedulers)]
    counts, handles = step_counters(optimizers)
    probe = Probe()
    trainer = make_trainer(model, wrapped, precision,
                           callbacks=[GradClipCallback(0.05, norm_type=2), probe])
    try:
        fit(trainer, DataLoader(TensorDataset(inputs, targets), batch_size=2))
        assert counts == [1, 1], counts
        assert len(probe.gradients) == 1
        gradient_error = assert_close(probe.gradients[0], expected_gradient, precision, gradient=True)
        weight_error = assert_close(parameters(model), parameters(reference), precision)
        assert [opt.param_groups[0]['lr'] for opt in optimizers] == [0.015, 0.035]
        if trainer.scaler.is_enabled():
            assert trainer.scaler.state_dict()['_growth_tracker'] == 1
        assert_rank_state(context, model, trainer, counts, schedulers)
        return {'optimizer_steps': counts, 'max_gradient_error': gradient_error,
                'max_weight_error': weight_error, 'scaler': trainer.scaler.state_dict()}
    finally:
        for handle in handles:
            handle.remove()
        trainer.close()


class TwoHeads(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.left = linear_model(device)
        self.right = linear_model(device)

    def forward(self, inputs, index):
        return self.left(inputs) if index == 0 else self.right(inputs)


class IndependentTask(EpochTask):
    def train_step(self, batch_x, batch_y):
        losses, predictions = [], []
        for index in (0, 1):
            output = self.model(batch_x, index=index)
            losses.append(self.loss(output, batch_y[:, index:index + 1],
                                    do_optimize=False, do_metric=False))
            self.optimize(optimize_index=index)
            predictions.append(output)
        self.do_metric(sum(losses) / 2, torch.cat(predictions, dim=1), batch_y)


def independent_optimizers_case(context, precision):
    inputs, first_targets = regression_data(context.world_size * 2)
    targets = torch.cat((first_targets, first_targets * 0.5 - 0.1), dim=1)
    reference = TwoHeads('cpu')
    reference_optimizers = [torch.optim.SGD(reference.left.parameters(), lr=0.03),
                            torch.optim.SGD(reference.right.parameters(), lr=0.07)]
    expected_gradients = []
    for index, optimizer in enumerate(reference_optimizers):
        optimizer.zero_grad()
        F.mse_loss(reference(inputs, index), targets[:, index:index + 1]).backward()
        expected_gradients.append(gradients(reference))
        optimizer.step()
        optimizer.zero_grad()

    model = TwoHeads(context.device)
    optimizers = [torch.optim.SGD(model.left.parameters(), lr=0.03),
                  torch.optim.SGD(model.right.parameters(), lr=0.07)]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5) for opt in optimizers]
    wrapped = [Optimizer(opt, scheduler, sched_on='step') for opt, scheduler in zip(optimizers, schedulers)]
    counts, handles = step_counters(optimizers)
    probe = Probe()
    trainer = make_trainer(model, wrapped, precision, callbacks=[probe],
                           ddp_kwargs={'find_unused_parameters': True})
    try:
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        fit(trainer, train_tasks=[IndependentTask(loader)])
        assert counts == [1, 1], counts
        assert len(probe.gradients) == 2
        gradient_error = max(assert_close(actual, expected, precision, gradient=True)
                             for actual, expected in zip(probe.gradients, expected_gradients))
        weight_error = assert_close(parameters(model), parameters(reference), precision)
        assert [opt.param_groups[0]['lr'] for opt in optimizers] == [0.015, 0.035]
        assert_rank_state(context, model, trainer, counts, schedulers)
        return {'optimizer_steps': counts, 'manual_optimizer_indices': [0, 1],
                'max_gradient_error': gradient_error, 'max_weight_error': weight_error,
                'scaler': trainer.scaler.state_dict()}
    finally:
        for handle in handles:
            handle.remove()
        trainer.close()


class TwoParameters(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.bad = nn.Parameter(torch.tensor(1., device=device))
        self.good = nn.Parameter(torch.tensor(1., device=device))

    def forward(self, inputs):
        return self.bad * inputs, self.good * inputs


def clipping_groups_case(context, precision, nonfinite=False):
    model = TwoParameters(context.device)
    optimizers = [torch.optim.SGD([parameter], lr=0.1) for parameter in (model.bad, model.good)]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5) for opt in optimizers]
    wrapped = [Optimizer(opt, scheduler, sched_on='step') for opt, scheduler in zip(optimizers, schedulers)]

    def loss(outputs, targets):
        first, second = outputs
        return ((first * float('nan')).mean() if nonfinite else first.square().mean()) + second.square().mean()

    counts, handles = step_counters(optimizers)
    trainer = make_trainer(model, wrapped, precision, callbacks=[GradClipCallback(1., norm_type=2)])
    trainer.loss.loss_fn = loss
    previous_scale = trainer.scaler.get_scale()
    try:
        loader = DataLoader(TensorDataset(torch.ones(context.world_size),
                                         torch.zeros(context.world_size)), batch_size=1)
        fit(trainer, loader)
        if nonfinite:
            assert precision == 'fp16' and trainer.scaler.is_enabled()
            torch.testing.assert_close(model.bad.detach().cpu(), torch.tensor(1.), atol=0, rtol=0)
            torch.testing.assert_close(model.good.detach().cpu(), torch.tensor(0.9), atol=1e-6, rtol=1e-6)
            assert counts == [0, 1], counts
            assert [scheduler.last_epoch for scheduler in schedulers] == [0, 1]
            assert [opt.param_groups[0]['lr'] for opt in optimizers] == [0.1, 0.05]
            assert trainer.scaler.get_scale() == previous_scale * 0.5
        else:
            expected = torch.tensor(1 - 0.1 / math.sqrt(2))
            for parameter in (model.bad, model.good):
                torch.testing.assert_close(parameter.detach().cpu(), expected, atol=1e-6, rtol=1e-6)
            assert counts == [1, 1], counts
            assert [scheduler.last_epoch for scheduler in schedulers] == [1, 1]
            assert [opt.param_groups[0]['lr'] for opt in optimizers] == [0.05, 0.05]
        assert_rank_state(context, model, trainer, counts, schedulers)
        return {'optimizer_steps': counts, 'weights': [model.bad.item(), model.good.item()],
                'scheduler_steps': [scheduler.last_epoch for scheduler in schedulers],
                'scaler': trainer.scaler.state_dict()}
    finally:
        for handle in handles:
            handle.remove()
        trainer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--expected-world-size', type=int, choices=(2, 3, 4), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    context = DistributedContext('cuda')
    records = []

    def run(name, function):
        details = function()
        context.barrier()
        record = {'case': name, 'status': 'PASS', 'precision': args.precision,
                  'world_size': context.world_size, **details}
        records.append(record)
        if context.is_main_process:
            print(json.dumps(record), flush=True)

    try:
        assert context.world_size == args.expected_world_size, (context.world_size, args.expected_world_size)
        if args.precision == 'bf16':
            assert torch.cuda.is_bf16_supported(), '当前 GPU 不支持 BF16。'
        run('cuda_device_binding_and_nccl', lambda: device_collectives(context))
        run('per_rank_global_gradient_reference', lambda: global_batch_case(context, args.precision, 4))
        run('split_global_batch_reference', lambda: global_batch_case(context, args.precision, 4, split=True))
        run('accumulation_tail_window', lambda: global_batch_case(context, args.precision, 5, accumulation=2))
        run('accumulation_unequal_microbatches', lambda: global_batch_case(context, args.precision, 3, accumulation=2))
        run('interrupted_no_sync_accumulation_retry',
            lambda: interrupted_accumulation_case(context, args.precision, 'batch'))
        run('interrupted_unscaled_update_retry',
            lambda: interrupted_accumulation_case(context, args.precision, 'optimize'))
        run('random_callback_parameter_and_buffer_sync', lambda: initialization_case(context, args.precision))
        run('shared_optimizers_unscale_and_clip', lambda: shared_optimizers_case(context, args.precision))
        run('independent_manual_optimizers', lambda: independent_optimizers_case(context, args.precision))
        run('finite_optimizer_joint_clipping', lambda: clipping_groups_case(context, args.precision))
        if args.precision == 'fp16':
            run('nonfinite_optimizer_skip_and_clipping',
                lambda: clipping_groups_case(context, args.precision, nonfinite=True))
        if context.is_main_process and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'torch': torch.__version__, 'precision': args.precision,
                                               'world_size': context.world_size, 'cases': records},
                                              indent=2) + '\n', encoding='utf-8')
    finally:
        context.close()


if __name__ == '__main__':
    main()
