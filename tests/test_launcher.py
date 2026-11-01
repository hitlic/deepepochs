"""launch入口的直接调用、真实Gloo进程、错误收尾与外部启动。"""
import json
import os
from pathlib import Path
import re
import shutil
import socket
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from deepepochs import launch


TESTS = Path(__file__).resolve().parent


def current_process():
    return {'pid': os.getpid(), 'initialized': dist.is_initialized()}


def failing_worker():
    directory = Path(os.environ['DEEPEPOCHS_LAUNCH_FAILURE_DIR'])
    rank = int(os.environ['RANK'])
    with (directory / f'rank-{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'omp': os.environ.get('OMP_NUM_THREADS')}) + '\n')
    dist.init_process_group('gloo', init_method='env://')
    dist.barrier()
    if rank == 1:
        raise RuntimeError('intentional launch worker failure')
    # rank0 stays alive until the launcher closes siblings after rank1 fails.
    while True:
        time.sleep(1)


def failure_run(directory):
    os.environ['DEEPEPOCHS_LAUNCH_FAILURE_DIR'] = directory
    before = os.environ.get('OMP_NUM_THREADS')
    signals = {getattr(signal, name): signal.getsignal(getattr(signal, name))
               for name in ('SIGINT', 'SIGTERM', 'SIGHUP', 'SIGQUIT') if hasattr(signal, name)}
    start = time.monotonic()
    try:
        launch(failing_worker, num_processes=2)
    except Exception as error:
        assert 'intentional launch worker failure' in str(error), str(error)
    else:
        raise AssertionError('worker failure did not reach launcher caller')
    assert os.environ.get('OMP_NUM_THREADS') == before
    assert all(signal.getsignal(number) == handler for number, handler in signals.items())
    workers = [json.loads(line) for path in sorted(Path(directory).glob('rank-*.jsonl'))
               for line in path.read_text().splitlines()]
    assert len(workers) == 2, workers  # No automatic retries.
    for worker in workers:
        try:
            os.kill(worker['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"worker is still alive: {worker['pid']}")
        assert worker['omp'] == before
    (Path(directory) / 'failure.json').write_text(json.dumps({
        'workers': workers, 'signals_restored': True, 'seconds': time.monotonic() - start,
        'parent_omp_before': before, 'parent_omp_after': os.environ.get('OMP_NUM_THREADS')}))


class LauncherTests(unittest.TestCase):
    def test_single_process_runs_in_caller_and_returns_result(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('RANK', None)
            result = launch(current_process, num_processes=1)
        self.assertEqual(result, {'pid': os.getpid(), 'initialized': False})

    def test_auto_without_cuda_uses_one_process(self):
        with patch('torch.cuda.is_available', return_value=False):
            result = launch(current_process)
        self.assertEqual(result['pid'], os.getpid())

    def test_auto_with_one_visible_gpu_uses_caller(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            result = launch(current_process)
        self.assertEqual(result['pid'], os.getpid())

    def test_invalid_entrypoint_and_process_count(self):
        with self.assertRaises(TypeError):
            launch(None, num_processes=1)
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                launch(current_process, num_processes=count)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要Gloo')
    def test_initialized_group_runs_directly_and_remains_owned_by_caller(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            os.environ.pop('RANK', None)
            store = (Path(directory) / 'store').resolve().as_uri()
            dist.init_process_group('gloo', rank=0, world_size=1, init_method=store)
            try:
                result = launch(current_process, num_processes=2)
                self.assertEqual(result, {'pid': os.getpid(), 'initialized': True})
                self.assertTrue(dist.is_initialized())
            finally:
                dist.destroy_process_group()

    def run_process(self, command, environment=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'deepepochs-launch-mpl'))
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'OMP_NUM_THREADS'):
            env.pop(key, None)
        env.update(environment or {})
        result = subprocess.run(command, cwd=TESTS.parent, env=env, capture_output=True, text=True, timeout=75)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-12000:])
        return result

    @unittest.skipUnless((TESTS.parent / 'examples' / 'deepepochs').is_symlink(), '需要checkout中的示例源码链接')
    def test_copied_examples_import_their_own_source(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            shutil.copytree(TESTS.parent / 'src' / 'deepepochs', snapshot / 'src' / 'deepepochs',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            examples = snapshot / 'examples'
            examples.mkdir()
            (examples / 'deepepochs').symlink_to(os.readlink(TESTS.parent / 'examples' / 'deepepochs'))
            probe = examples / 'import_probe.py'
            probe.write_text('import deepepochs\nprint(deepepochs.__file__)\n')
            result = self.run_process([sys.executable, str(probe)],
                                      environment={'PYTHONPATH': str(snapshot / 'src')})
            self.assertEqual(Path(result.stdout.strip()).resolve(),
                             (snapshot / 'src' / 'deepepochs' / '__init__.py').resolve())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要Gloo')
    def test_readme_example_trains_with_three_cpu_workers(self):
        readme = (TESTS.parent / 'README.md').read_text()
        examples = re.findall(r'^```python\n(.*?)^```', readme, re.M | re.S)
        code = next(example for example in examples if 'def main():' in example)
        code = code.replace('results = launch(main)', 'results = launch(main, num_processes=3)')
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'readme_launch.py'
            script.write_text(code + "    assert isinstance(results, dict) and 'loss' in results\n")
            result = self.run_process([sys.executable, str(script)])
        self.assertIn('processes:   3 (DDP)', result.stdout)
        self.assertIn('E 5/5', result.stdout)
        self.assertIn('TEST>', result.stdout)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要Gloo')
    def test_two_cpu_workers_train_return_rank_zero_and_restore_default_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_process([sys.executable, str(TESTS / 'smoke_launch.py'), '--num-processes', '2',
                              '--expected-world-size', '2', '--device', 'cpu', '--output', directory])
            result = json.loads((Path(directory) / 'launch.json').read_text())
        self.assertEqual(result['world_size'], 2)
        self.assertEqual(result['rank'], 0)
        self.assertNotEqual(result['launcher_parent_pid'], result['per_rank'][0]['pid'])
        self.assertEqual(result['parent_omp_before'], None)
        self.assertEqual(result['parent_omp_after'], None)
        self.assertEqual([item['omp_num_threads'] for item in result['per_rank']], ['1', '1'])
        self.assertTrue(result['writer_closed'] and result['process_group_closed'])
        self.assertTrue(result['global_metric_reference'])
        self.assertTrue(result['signals_restored'])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要Gloo')
    def test_torchrun_workers_do_not_start_nested_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            self.run_process([sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1',
                              f'--master-port={port}', '--nproc-per-node=2', str(TESTS / 'smoke_launch.py'),
                              '--num-processes', '4', '--expected-world-size', '2', '--device', 'cpu',
                              '--output', directory])
            result = json.loads((Path(directory) / 'launch.json').read_text())
        self.assertEqual(result['world_size'], 2)
        self.assertEqual(result['launcher_parent_pid'], result['per_rank'][0]['pid'])

    @unittest.skipUnless(os.name == 'posix' and dist.is_available() and dist.is_gloo_available(), '需要POSIX/Gloo')
    def test_worker_failure_stops_siblings_without_retry_and_preserves_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_process([sys.executable, str(TESTS / 'test_launcher.py'), '--failure-run', directory],
                             environment={'OMP_NUM_THREADS': '3'})
            result = json.loads((Path(directory) / 'failure.json').read_text())
        self.assertEqual(len(result['workers']), 2)
        self.assertTrue(result['signals_restored'])
        self.assertEqual(result['parent_omp_before'], '3')
        self.assertEqual(result['parent_omp_after'], '3')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--failure-run':
        failure_run(sys.argv[2])
    else:
        unittest.main()
