"""真实终端与文件中的默认训练布局，以及恢复后的启动信息。"""
import contextlib
import errno
import io
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import deepepochs
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepepochs import Trainer
from deepepochs.callbacks.check import save_state


PROGRAM = '''
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from deepepochs import CheckCallback, LogCallback, Trainer
from deepepochs.callbacks.check import save_state

torch.set_num_threads(1)
torch.manual_seed(7)
inputs = torch.arange(4.).reshape(-1, 1) / 4
loader = DataLoader(TensorDataset(inputs, inputs * 2), batch_size=2)
model = nn.Linear(1, 1)
def mae(predictions, targets):
    return (predictions - targets).abs().mean()
checks = [CheckCallback('loss', ckpt_dir='checkpoints'),
          CheckCallback('mae', save_best=False, ckpt_dir='checkpoints')]
logging = LogCallback(log_dir='metrics')
with patch('deepepochs.callbacks.check.save_state', wraps=save_state) as saves, \
        Trainer(model, nn.functional.mse_loss, torch.optim.SGD(model.parameters(), lr=.01),
             device='cpu', epochs=2, metrics=[mae], log_tqdm=sys.argv[1] == '1',
             log_batch=sys.argv[2] == '1', callbacks=[*checks, logging],
             gradient_accumulation_steps=int(sys.argv[3]) if len(sys.argv) > 3 else 1) as trainer:
    trainer.fit(loader, loader)
    assert sorted(Counter(call.args[3] for call in saves.call_args_list).values()) == [2, 2]
    assert all(torch.load(check.ckpt_path, weights_only=False)['epoch'] == 1 for check in checks)
    assert Path(logging.logger.log_dir).is_dir()
    trainer.test(loader, weights='current')
'''
STYLE = re.compile(r'\x1b\[[0-9;]*m')

WIDE_PROGRAM = """
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from deepepochs import Callback, EpochTask, Trainer
from deepepochs import metrics

torch.set_num_threads(1)
torch.manual_seed(7)
def acc(predictions, targets):
    return metrics.accuracy(predictions, targets)
def r(predictions, targets):
    return metrics.recall(predictions, targets)
class Notice(Callback):
    def __init__(self):
        super().__init__(priority=-1)
    def snapshot(self):
        # BEL不占显示列，供测试在每批刷新完成后回放可见屏幕。
        print(chr(7), end='', flush=True)
    def on_after_train_batch(self, trainer, metrics, batch_idx):
        self.snapshot()
        if batch_idx == 1:
            trainer.print('[NOTICE] batch complete')
    def on_after_val_batch(self, trainer, metrics, batch_idx):
        self.snapshot()

loader = DataLoader(TensorDataset(torch.tensor([[1., 0.]]).repeat(6, 1),
                                 torch.zeros(6, dtype=torch.long)), batch_size=2)
model = nn.Linear(2, 2)
with Trainer(model, nn.functional.cross_entropy, torch.optim.SGD(model.parameters(), lr=.01),
             device='cpu', epochs=2, callbacks=[Notice()], show_info=False,
             log_tqdm=sys.argv[1] == '1') as trainer:
    trainer.fit(train_tasks=[EpochTask(loader, metrics=[acc, r], name='accuracy_task')],
                val_tasks=[EpochTask(loader, metrics=[acc, r], name='accuracy_task')])
"""


def terminal_lines(text, columns):
    """回放本测试使用的光标控制和延迟自动换行，检查实际可见内容。"""
    screen = [[' '] * columns]
    row = column = 0
    for token in re.findall(r'\x1b\[[0-9;]*[A-Za-z]|.', text, re.DOTALL):
        if token.startswith('\x1b['):
            command = token[-1]
            argument = int(token[2:-1] or '0') if command != 'm' else 0
            if command == 'A':
                row = max(0, row - (argument or 1))
            elif command == 'K':
                start = 0 if argument == 2 else min(column, columns)
                screen[row][start:] = [' '] * (columns - start)
            elif command != 'm':
                raise AssertionError(f'Unexpected terminal control: {token!r}')
        elif token == '\a':
            continue
        elif token == '\r':
            column = 0
        elif token == '\n':
            row += 1
            if row == len(screen):
                screen.append([' '] * columns)
        else:
            if column == columns:
                row += 1
                column = 0
                if row == len(screen):
                    screen.append([' '] * columns)
            screen[row][column] = token
            column += 1
    return [''.join(line).rstrip() for line in screen]


class ConsoleIntegrationTests(unittest.TestCase):
    def environment(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', TERM='xterm',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   PYTHONPATH=str(Path(deepepochs.__file__).resolve().parent.parent))
        env.pop('NO_COLOR', None)
        return env

    def assert_summaries(self, lines):
        epochs = [line for line in lines if re.match(r'E\s+\d+/2\s+TRAIN>', line)]
        self.assertEqual(len(epochs), 2, lines)
        for line in epochs:
            self.assertIn('TRAIN> loss:', line)
            self.assertIn('VAL> loss:', line)
            self.assertEqual(line.count('mae:'), 2)
            self.assertNotIn('lr:', line)
            self.assertNotIn('time:', line)
            self.assertRegex(line, r'\S {3}\[\d+\.\d{2}s\]$')
        testing = [line for line in lines if re.match(r'E\s+1/1\s+TEST>', line)]
        self.assertEqual(len(testing), 1, lines)
        self.assertIn('loss:', testing[0])
        self.assertIn('mae:', testing[0])
        self.assertNotIn('lr:', testing[0])
        self.assertNotIn('time:', testing[0])
        self.assertRegex(testing[0], r'\S {3}\[\d+\.\d{2}s\]$')

        first, last = (lines.index(line) for line in epochs)
        between_epochs = lines[first + 1:last]
        self.assertTrue(all(re.match(r'E\s+2/2\s+B\s+\d+/\d+\s+(TRAIN|VAL)>', line)
                            for line in between_epochs), between_epochs)
        for label in ('BEST', 'LATEST', 'LOG'):
            summaries = [line for line in lines if line.startswith(f'[{label}]')]
            self.assertEqual(len(summaries), 1, lines)
            self.assertGreater(lines.index(summaries[0]), last)
            self.assertLess(lines.index(summaries[0]), lines.index(testing[0]))

    def run_in_terminal(self, program, arguments=(), columns=240, no_color=False):
        import fcntl
        import pty
        import struct
        import termios

        with tempfile.TemporaryDirectory() as directory:
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 60, columns, 0, 0))
            env = self.environment()
            # 故意与实际列宽不同，确认刷新读取当前终端而非环境变量。
            env['COLUMNS'] = '240'
            if no_color:
                env['NO_COLOR'] = '1'
            process = subprocess.Popen([sys.executable, '-c', program, *arguments],
                                       cwd=directory, env=env, stdout=slave, stderr=slave)
            os.close(slave)
            chunks = []
            deadline = time.monotonic() + 90
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([master], [], [], remaining)[0]:
                        self.fail('Training did not finish in the terminal')
                    try:
                        data = os.read(master, 8192)
                    except OSError as error:
                        if error.errno != errno.EIO:
                            raise
                        break
                    if not data:
                        break
                    chunks.append(data)
                process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                os.close(master)
            output = b''.join(chunks).decode(errors='replace')
            self.assertEqual(process.returncode, 0, output)
            return output

    @unittest.skipUnless(os.name == 'posix', 'requires a POSIX terminal')
    def test_real_terminal_preserves_batch_refresh_and_single_line_epoch_summaries(self):
        for use_tqdm, accumulation in ((False, 1), (True, 1), (False, 2)):
            with self.subTest(tqdm=use_tqdm, accumulation=accumulation):
                output = self.run_in_terminal(PROGRAM, [str(int(use_tqdm)), '1', str(accumulation)])
                self.assertIn('\x1b[', output)
                plain = STYLE.sub('', output)
                self.assert_summaries(terminal_lines(output, 240))
                self.assertIn('\x1b[36mTRAIN\x1b[0m', output)
                self.assertIn('\x1b[33mVAL\x1b[0m', output)
                self.assertEqual(len(re.findall(r' {3}\x1b\[90m\[\d+\.\d{2}s\]\x1b\[0m\r?\n', output)), 4)
                self.assertEqual(plain.count('init lr:'), 1)
                self.assertIn('precision:', plain)
                accumulation_info = [line for line in plain.splitlines() if line.startswith('accumulate:')]
                if accumulation > 1:
                    self.assertEqual(len(accumulation_info), 1)
                    self.assertEqual(accumulation_info[0].split()[-1], '2')
                else:
                    self.assertEqual(accumulation_info, [])
                self.assertIn('[FINISHED] epochs: 2', plain)
                if not use_tqdm:
                    self.assertEqual(len(re.findall(r'\bB\s+\d+/\d+', plain)), 10)
                self.assertLess(plain.index('parameters:'), plain.index('B 1/2'))

    @unittest.skipUnless(os.name == 'posix', 'requires a POSIX terminal')
    def test_real_80_column_terminal_clears_wrapped_named_batch_metrics(self):
        for use_tqdm in (False, True):
            for no_color in (False, True):
                with self.subTest(tqdm=use_tqdm, no_color=no_color):
                    output = self.run_in_terminal(WIDE_PROGRAM, [str(int(use_tqdm))],
                                                  columns=80, no_color=no_color)
                    # 检查每次逐批刷新后的屏幕，避免仅靠轮末完整指标掩盖批次截断。
                    parts = output.split('\a')
                    self.assertEqual(len(parts), 13, output)
                    prefix = ''
                    for part in parts[:-1]:
                        prefix += part
                        visible = ''.join(line.ljust(80) for line in terminal_lines(prefix, 80))
                        batches = list(re.finditer(r'\bE\s+\d+/2\s+B\s+\d+/3', visible))
                        self.assertTrue(batches, visible)
                        current = visible[batches[-1].start():]
                        for key in ('loss', 'acc', 'r'):
                            self.assertRegex(current, rf'accuracy_task/{key}:\s+\d+\.\d{{4}}', current)
                    lines = terminal_lines(output, 80)
                    visible = ''.join(lines)
                    self.assertNotRegex(visible, r'\bB\s+\d+/3')
                    summaries = [line for line in lines if re.match(r'E\s+\d+/2\s+TRAIN>', line)]
                    self.assertEqual(len(summaries), 2, lines)
                    for key in ('accuracy_task/loss:', 'accuracy_task/acc:', 'accuracy_task/r:'):
                        self.assertEqual(visible.count(key), 4, lines)
                    self.assertEqual(visible.count('[NOTICE] batch complete'), 2, lines)
                    self.assertEqual(bool(STYLE.search(output)), not no_color)

    @unittest.skipUnless(os.name == 'posix', 'requires a POSIX terminal')
    def test_zero_column_terminal_uses_complete_line_output(self):
        for use_tqdm in (False, True):
            with self.subTest(tqdm=use_tqdm):
                output = self.run_in_terminal(PROGRAM, [str(int(use_tqdm)), '1'],
                                              columns=0, no_color=True)
                # PTY将普通换行转换为CRLF；除此之外不应有动态刷新控制符。
                plain = output.replace('\r\n', '\n')
                self.assertNotIn('\r', plain)
                self.assertNotIn('\x1b', plain)
                self.assert_summaries(plain.splitlines())
                batches = [line for line in plain.splitlines() if re.search(r'\bB\s+\d+/\d+', line)]
                self.assertEqual(len(batches), 10)
                self.assertTrue(all('loss:' in line and 'mae:' in line for line in batches))
                self.assertIn('[FINISHED]', plain)

    def test_redirected_output_is_plain_lines_and_respects_batch_logging(self):
        for use_tqdm, log_batch in ((False, True), (True, True), (False, False)):
            with self.subTest(tqdm=use_tqdm, batches=log_batch), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, '-c', PROGRAM, str(int(use_tqdm)), str(int(log_batch))],
                                        cwd=directory, env=self.environment(), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('\x1b[', result.stdout)
                self.assertNotIn('\r', result.stdout)
                self.assert_summaries(result.stdout.splitlines())
                self.assertEqual(result.stdout.count('init lr:'), 1)
                batches = [line for line in result.stdout.splitlines() if re.search(r'\bB\s+\d+/\d+', line)]
                self.assertEqual(len(batches), 10 if log_batch else 0)
                if log_batch:
                    self.assertTrue(all('loss:' in line and 'mae:' in line for line in batches))

    def test_startup_uses_loaded_state_and_checkpoint_message_is_main_process_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.ckpt'
            source = nn.Linear(1, 1)
            with Trainer(source, nn.functional.mse_loss,
                         torch.optim.SGD(source.parameters(), lr=.05), device='cpu', show_info=False) as trainer:
                save_state(trainer, trainer.model, trainer.opt, path, epoch=1)
            model = nn.Linear(1, 1)
            loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1)
            output = io.StringIO()
            with Trainer(model, nn.functional.mse_loss, torch.optim.SGD(model.parameters(), lr=.8),
                         device='cpu', epochs=3, log_batch=False) as trainer:
                with contextlib.redirect_stdout(output):
                    trainer.load_checkpoint(path)
                    trainer.fit(loader)
                lines = output.getvalue().splitlines()
                self.assertIn('3..3', next(line for line in lines if line.startswith('epochs:')))
                self.assertIn('5.000e-02', next(line for line in lines if line.startswith('init lr:')))
                self.assertEqual(output.getvalue().count('[LOADED]'), 1)
                trainer.main_process = False
                silent = io.StringIO()
                with patch('deepepochs.callbacks.check.load_state'), contextlib.redirect_stdout(silent):
                    trainer.load_checkpoint(path)
                self.assertEqual(silent.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
