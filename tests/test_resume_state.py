"""重建训练进程后的 epoch 边界状态与连续训练一致。"""
import contextlib
from copy import deepcopy
import io
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from deepepochs import Callback, CheckCallback, EpochTask, Optimizer, Trainer
from deepepochs.callbacks.check import (
    collect_rank_state, finalize_checkpoints, load_evaluation_model, load_state,
    restore_fit_random_state, restore_fit_state, restore_test_model, save_state,
)


class RandomDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return torch.tensor([index / 12. + noise]), torch.tensor([index / 6.])


class ResettingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.resource = object()

    def on_before_fit(self, trainer, epochs):
        self.count = 0
        random.random()
        np.random.rand()
        torch.rand(())
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.fill_(0.25)

    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.count += 1

    def state_dict(self):
        return {'count': self.count}

    def load_state_dict(self, state):
        self.count = state['count']



class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.register_buffer('marker', torch.tensor(9.))

    def forward(self, inputs):
        return inputs * self.weight


class ResetModelAndOptimizer(Callback):
    def on_before_fit(self, trainer, epochs):
        with torch.no_grad():
            trainer.model.weight.fill_(1.)
            trainer.model.marker.fill_(9.)
        optimizer = trainer.opt[0]
        optimizer.opt.state.clear()
        optimizer.param_groups[0]['lr'] = 0.1
        optimizer.scheduler.last_epoch = 0

def _rank_resume_worker(rank, rendezvous, directory):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    trainers = []

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def make_trainer(root, running_id, resume=False):
        model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Dropout(0.3), nn.Linear(4, 1))
        callback = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=root)
        trainer = Trainer(model, nn.functional.mse_loss,
                          torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8),
                          callbacks=[callback, ResettingCallback()], device='cpu',
                          show_info=False, log_batch=False, running_id=running_id, resume=resume)
        trainers.append(trainer)
        return trainer, callback

    def loader(value):
        return DataLoader(RandomDataset(), batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(value + rank))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            seed(20 + rank)
            full, _ = make_trainer(str(Path(directory) / 'full'), 'full')
            full_loader = loader(80)
            full.fit(full_loader, epochs=3)
            expected_model = deepcopy(full.model.state_dict())
            expected_random = (random.random(), float(np.random.rand()), torch.rand(()))
            expected_generator = full.train_tasks[0]._train_dataloader.generator.get_state()

            seed(20 + rank)
            source, callback = make_trainer(str(Path(directory) / 'split'), 'saved')
            source.fit(loader(80), epochs=1)
            if rank == 0:
                state = torch.load(callback.ckpt_path, weights_only=False)
                assert len(state['rank_states']) == 2
                assert not torch.equal(state['rank_states'][0]['random']['torch'],
                                       state['rank_states'][1]['random']['torch'])
            for entry in ('automatic', 'explicit'):
                seed(900 + rank)
                resumed, _ = make_trainer(str(Path(directory) / 'split'), entry,
                                          resume='saved' if entry == 'automatic' else False)
                if entry == 'explicit':
                    resumed.load_checkpoint(callback.ckpt_path)
                resumed.fit(loader(700), epochs=3)
                for name, value in resumed.model.state_dict().items():
                    torch.testing.assert_close(value, expected_model[name], atol=0, rtol=0)
                assert (random.random(), float(np.random.rand())) == expected_random[:2]
                torch.testing.assert_close(torch.rand(()), expected_random[2], atol=0, rtol=0)
                torch.testing.assert_close(resumed.train_tasks[0]._train_dataloader.generator.get_state(),
                                           expected_generator, atol=0, rtol=0)
    finally:
        for trainer in trainers:
            trainer.close()
        dist.destroy_process_group()


class ResumeStateTests(unittest.TestCase):
    def seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def trainer(self, callbacks=None, **kwargs):
        model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Dropout(0.3), nn.Linear(4, 1))
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
        trainer = Trainer(model, nn.functional.mse_loss, Optimizer(opt, scheduler), callbacks=callbacks,
                          device='cpu', show_info=False, log_batch=False, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def loader(self, seed=81):
        return DataLoader(RandomDataset(), batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_callback_tensor_state_is_an_independent_cpu_snapshot(self):
        class TensorState(Callback):
            def __init__(self):
                super().__init__()
                self.value = torch.tensor([1., 2.], requires_grad=True)

            def state_dict(self):
                return {'value': self.value, 'nested': [self.value.square(), None]}

        callback = TensorState()
        with Trainer(nn.Linear(1, 1), callbacks=[callback], device='cpu',
                     show_info=False, log_batch=False) as trainer:
            state = collect_rank_state(trainer)['callbacks']
        saved = next(item['state'] for item in state if item['identity'][0].endswith('TensorState'))
        self.assertEqual(saved['value'].device.type, 'cpu')
        self.assertFalse(saved['value'].requires_grad)
        self.assertIsNone(saved['nested'][0].grad_fn)
        self.assertIsNone(saved['nested'][1])
        self.assertNotEqual(saved['value'].data_ptr(), callback.value.data_ptr())
        with torch.no_grad():
            callback.value.add_(10.)
        torch.testing.assert_close(saved['value'], torch.tensor([1., 2.]))
        torch.testing.assert_close(saved['nested'][0], torch.tensor([1., 4.]))

    def test_dropout_shuffle_rng_initializer_and_callback_resume_match_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            self.seed(19)
            full_counter = ResettingCallback()
            full = self.trainer([CheckCallback('loss', on_stage='train', save_best=False,
                                               ckpt_dir=str(Path(directory) / 'full')), full_counter])
            full_loader = self.loader()
            full_progress = self.quiet(full.fit, full_loader, epochs=4)
            expected_model = deepcopy(full.model.state_dict())
            expected_opt = deepcopy(full.opt.state_dict())
            expected_random = (random.random(), float(np.random.rand()), torch.rand(()))
            expected_generator = full_loader.generator.get_state()

            self.seed(19)
            source_check = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=directory)
            source = self.trainer([source_check, ResettingCallback()], running_id='saved')
            self.quiet(source.fit, self.loader(), epochs=2)
            state = torch.load(source_check.ckpt_path, weights_only=False)
            saved_counter = next(item for item in state['rank_states'][0]['callbacks']
                                 if item['identity'][0].endswith('ResettingCallback'))
            self.assertEqual(saved_counter['state'], {'count': 8})
            self.seed(991)
            resumed_counter = ResettingCallback()
            resumed = self.trainer([CheckCallback('loss', on_stage='train', save_best=False,
                                                  ckpt_dir=directory), resumed_counter],
                                   running_id='continued', resume='saved')
            resumed_loader = self.loader(seed=999)
            resumed_progress = self.quiet(resumed.fit, resumed_loader, epochs=4)
            self.assert_nested_equal(resumed.model.state_dict(), expected_model)
            self.assert_nested_equal(resumed.opt.state_dict(), expected_opt)
            self.assertEqual(resumed_progress['train']['loss'], full_progress['train']['loss'][2:])
            self.assertEqual(resumed_counter.count, full_counter.count)
            self.assertEqual((random.random(), float(np.random.rand())), expected_random[:2])
            torch.testing.assert_close(torch.rand(()), expected_random[2], atol=0, rtol=0)
            torch.testing.assert_close(resumed_loader.generator.get_state(), expected_generator, atol=0, rtol=0)

    def test_explicit_load_preserves_user_model_changes_when_fit_restores_callback_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source_counter = ResettingCallback()
            source = self.trainer([source_counter])
            source_counter.count = 7
            path = Path(directory) / 'state.ckpt'
            save_state(source, source.model, source.opt, path, epoch=2)
            target_counter = ResettingCallback()
            target = self.trainer([target_counter])
            load_state(target, path)
            self.assertEqual(target.init_epoch, 3)
            with torch.no_grad():
                next(target.model.parameters()).fill_(4.)
            expected = deepcopy(target.model.state_dict())
            target_counter.count = 0
            restore_fit_state(target, [], [])
            self.assert_nested_equal(target.model.state_dict(), expected)
            self.assertEqual(target_counter.count, 7)

    def scalar_trainer(self, directory, running_id, resume=False, momentum=0.):
        model = ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)
        checker = CheckCallback('loss', on_stage='train', save_best=False, ckpt_dir=directory)
        trainer = Trainer(model, nn.functional.mse_loss, Optimizer(optimizer, scheduler),
                          callbacks=[checker, ResetModelAndOptimizer()], resume=resume,
                          running_id=running_id, device='cpu', epochs=2,
                          show_info=False, log_batch=False)
        self.addCleanup(trainer.close)
        return trainer, checker

    def test_explicit_and_automatic_resume_apply_state_after_initialization(self):
        loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            source, checker = self.scalar_trainer(directory, 'saved')
            self.quiet(source.fit, loader, epochs=1)
            self.assertAlmostEqual(float(source.model.weight.detach()), 0.8, places=6)
            automatic, _ = self.scalar_trainer(directory, 'automatic', resume='saved')
            self.quiet(automatic.fit, loader)
            explicit, _ = self.scalar_trainer(directory, 'explicit')
            self.quiet(explicit.load_checkpoint, checker.ckpt_path)
            self.assertAlmostEqual(float(explicit.model.weight.detach()), 0.8, places=6)
            self.quiet(explicit.fit, loader)
            self.assertAlmostEqual(float(explicit.model.weight.detach()), 0.64, places=6)
            self.assert_nested_equal(explicit.model.state_dict(), automatic.model.state_dict())
            self.assert_nested_equal(explicit.opt.state_dict(), automatic.opt.state_dict())
            self.assertEqual(explicit.init_epoch, automatic.init_epoch)

    def test_explicit_resume_keeps_user_parameter_buffer_and_optimizer_changes(self):
        loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            source, checker = self.scalar_trainer(directory, 'saved', momentum=0.9)
            self.quiet(source.fit, loader, epochs=1)
            target, _ = self.scalar_trainer(directory, 'explicit', momentum=0.9)
            self.quiet(target.load_checkpoint, checker.ckpt_path)
            optimizer = target.opt[0]
            with torch.no_grad():
                target.model.weight.fill_(2.)
                target.model.marker.fill_(7.)
                optimizer.opt.state[target.model.weight]['momentum_buffer'].fill_(3.)
            optimizer.param_groups[0]['lr'] = 0.05
            optimizer.scheduler.last_epoch = 17

            reference = deepcopy(target.model.model)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
            reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, 1, gamma=1.)
            reference_optimizer.load_state_dict(deepcopy(optimizer.opt.state_dict()))
            reference_scheduler.load_state_dict(deepcopy(optimizer.scheduler.state_dict()))
            inputs, labels = next(iter(loader))
            nn.functional.mse_loss(reference(inputs), labels).backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            reference_scheduler.step()

            self.quiet(target.fit, loader)
            self.assert_nested_equal(target.model.state_dict(), reference.state_dict())
            self.assert_nested_equal(optimizer.opt.state_dict(), reference_optimizer.state_dict())
            self.assert_nested_equal(optimizer.scheduler.state_dict(), reference_scheduler.state_dict())
            self.assertAlmostEqual(float(target.model.weight.detach()), 1.665, places=6)
            self.assertEqual(float(target.model.marker), 7.)
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.05)
            self.assertEqual(optimizer.scheduler.last_epoch, 18)

    def test_rank_specific_prepared_loader_generator_sampler_seed_and_rng_restore(self):
        source = self.trainer()
        source.world_size = 2
        states = []
        dataset = TensorDataset(torch.arange(8))
        for rank in range(2):
            self.seed(40 + rank)
            loader = DataLoader(dataset, batch_size=2,
                                sampler=DistributedSampler(dataset, num_replicas=2, rank=rank, seed=60 + rank),
                                generator=torch.Generator().manual_seed(80 + rank))
            task = EpochTask(loader)
            task._train_dataloader = loader
            source.train_tasks, source.val_tasks = [task], []
            states.append(collect_rank_state(source))
        target = self.trainer()
        target.rank, target.world_size = 1, 2
        actual = DataLoader(dataset, batch_size=2,
                            sampler=DistributedSampler(dataset, num_replicas=2, rank=1, seed=0),
                            generator=torch.Generator().manual_seed(0))
        unused = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(12))
        task = EpochTask(unused)
        task._train_dataloader = actual
        state = {'model_state': deepcopy(source.model.state_dict()), 'opt_state': source.opt.state_dict(),
                 'checkpoint_file': 'state.ckpt', 'callback_states': [], 'epoch': 1,
                 'world_size': 2, 'rank_states': states}
        target._pending_fit_state = (state, '/state.ckpt', True)
        restore_fit_state(target, [task], [])
        self.assertEqual(actual.sampler.seed, 61)
        self.assertEqual(unused.generator.initial_seed(), 12)
        torch.testing.assert_close(actual.generator.get_state(), states[1]['loaders'][0]['state']['loader']['generator'])
        restore_fit_random_state(target)
        self.assertEqual(random.getstate(), states[1]['random']['python'])
        self.assertTrue(np.array_equal(np.random.get_state()[1], states[1]['random']['numpy'][1]))
        torch.testing.assert_close(torch.get_rng_state(), states[1]['random']['torch'], atol=0, rtol=0)

    def test_world_size_change_loads_training_state_without_rank_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.trainer()
            path = Path(directory) / 'state.ckpt'
            save_state(source, source.model, source.opt, path, epoch=2)
            target = self.trainer()
            target.world_size = 2
            load_state(target, path)
            rng = torch.get_rng_state()
            restore_fit_state(target, [], [])
            restore_fit_random_state(target)
            self.assertEqual(target.init_epoch, 3)
            self.assert_nested_equal(target.model.state_dict(), source.model.state_dict())
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)

    def test_finalize_collects_once_for_all_monitor_files_and_on_non_main_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            callbacks = [CheckCallback('loss', on_stage='train', ckpt_dir=directory) for _ in range(2)]
            trainer = self.trainer(callbacks)
            for callback in callbacks:
                callback.do_check(trainer, {'loss': 1.}, 0, defer=True)
            with patch.object(trainer.distributed, 'all_gather_object', wraps=trainer.distributed.all_gather_object) as gather:
                finalize_checkpoints(trainer)
            self.assertEqual(gather.call_count, 1)
            left, right = [torch.load(callback.ckpt_path, weights_only=False) for callback in callbacks]
            self.assert_nested_equal(left['rank_states'][0]['random']['torch'], right['rank_states'][0]['random']['torch'])
            trainer.main_process = False
            trainer._pending_checkpoints = [(callbacks[0], True, True, 1, 'unused')]
            decision = (None, None, [callback.ckpt_path for callback in callbacks])
            with patch.object(trainer.distributed, 'all_gather_object', return_value=[(collect_rank_state(trainer), None)]) as gather:
                with patch.object(trainer.distributed, 'broadcast_object', return_value=decision):
                    with patch('deepepochs.callbacks.check.save_state') as save:
                        finalize_checkpoints(trainer)
            gather.assert_called_once()
            save.assert_not_called()

    def test_two_rank_dropout_shuffle_resume_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_rank_resume_worker, args=(str(Path(directory) / 'rendezvous'), directory),
                     nprocs=2, join=True)

    def test_evaluation_weight_selection_restores_current_model(self):
        with tempfile.TemporaryDirectory() as directory:
            callbacks = [CheckCallback('loss', on_stage='train', ckpt_dir=directory) for _ in range(2)]
            trainer = self.trainer(callbacks)
            for index, callback in enumerate(callbacks):
                with torch.no_grad():
                    next(trainer.model.parameters()).fill_(index + 1.)
                callback.do_check(trainer, {'loss': 1.}, 0)
            with torch.no_grad():
                next(trainer.model.parameters()).fill_(3.)
            current = deepcopy(trainer.model.state_dict())
            for choice, expected in [('auto', 1.), ('current', 3.), (callbacks[1], 2.),
                                     (Path(callbacks[0].ckpt_path), 1.)]:
                load_evaluation_model(trainer, choice)
                self.assertEqual(float(next(trainer.model.parameters()).flatten()[0]), expected)
                trainer.callbacks.trigger('before_test_epochs', trainer=trainer, tasks=[])
                self.assertEqual(float(next(trainer.model.parameters()).flatten()[0]), expected)
                restore_test_model(trainer)
                self.assert_nested_equal(trainer.model.state_dict(), current)


if __name__ == '__main__':
    unittest.main()
