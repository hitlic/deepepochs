"""终端逐批刷新、轮末历史、纯文本重定向和数值显示。"""
import contextlib
import io
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tqdm import tqdm

from deepepochs.loops import (
    _ConsoleProgress, _clear_progress_line, _color, _format_learning_rates,
    info, log_batch, log_epoch, print_out,
)


class TerminalStream(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 1


def displayed_lines(text, columns=80):
    """回放自动换行、颜色和清行/上移控制，保留终端可见内容。"""
    screen, row, column, wrap_pending = [[' '] * columns], 0, 0, False
    offset = 0
    while offset < len(text):
        escape = re.match(r'\x1b\[([0-9;]*)([A-Za-z])', text[offset:])
        if escape:
            argument, command = escape.groups()
            if command == 'A':
                row = max(0, row - int(argument or 1))
                wrap_pending = False
            elif command == 'K':
                screen[row] = [' '] * columns
            offset += escape.end()
            continue
        char = text[offset]
        offset += 1
        if char == '\r':
            column, wrap_pending = 0, False
        elif char == '\n':
            row, column, wrap_pending = row + 1, 0, False
        else:
            if wrap_pending:
                row, column, wrap_pending = row + 1, 0, False
            while len(screen) <= row:
                screen.append([' '] * columns)
            screen[row][column] = char
            if column == columns - 1:
                wrap_pending = True
            else:
                column += 1
        while len(screen) <= row:
            screen.append([' '] * columns)
    lines = [''.join(line).rstrip() for line in screen]
    while lines and not lines[-1]:
        lines.pop()
    return lines


class ConsoleRenderingTests(unittest.TestCase):
    def setUp(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _clear_progress_line()

    tearDown = setUp

    @contextlib.contextmanager
    def output(self, terminal=False, no_color=False, term='xterm', columns=80):
        stream = TerminalStream() if terminal else io.StringIO()
        environment = dict(os.environ, TERM=term)
        environment.pop('NO_COLOR', None)
        if no_color:
            environment['NO_COLOR'] = ''
        with contextlib.redirect_stdout(io.StringIO()):
            _clear_progress_line()
        with patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(stream), \
                patch('deepepochs.loops.os.get_terminal_size', return_value=os.terminal_size((columns, 24))):
            yield stream

    def test_redirected_batches_and_epoch_are_independent_plain_text_lines(self):
        with self.output() as stream:
            log_batch({'loss': .4, 'accuracy': .9}, 1, 2, 1, 2, 'TRAIN', 1, 1)
            log_batch({'loss': .3, 'accuracy': 1.}, 1, 2, 2, 2, 'TRAIN', 1, 1)
            log_epoch({'train': {'loss': .35}, 'val': {'loss': .2}}, 1, 2, 1,
                      extra_info='[1.20s]')
        text = stream.getvalue()
        self.assertNotIn('\r', text)
        self.assertNotIn('\x1b', text)
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('B 1/2', lines[0])
        self.assertIn('accuracy: 0.9000', lines[0])
        self.assertIn('B 2/2', lines[1])
        self.assertIn('TRAIN> loss: 0.3500', lines[2])
        self.assertIn('VAL> loss: 0.2000', lines[2])
        self.assertTrue(lines[2].endswith('0.2000   [1.20s]'))
        self.assertNotIn('lr:', lines[2])
        self.assertNotIn('time:', lines[2])

    def test_terminal_updates_every_batch_and_replaces_the_longer_line(self):
        with self.output(terminal=True) as stream:
            log_batch({'loss': .4, 'a_long_metric_name': .9}, 1, 2, 1, 2, 'TRAIN', 1, 1)
            log_batch({'loss': .3}, 1, 2, 2, 2, 'TRAIN', 1, 1)
            updates = stream.getvalue()
            self.assertEqual(updates.count('E 1/2'), 2)
            self.assertNotIn('\n', updates)
            self.assertNotIn('a_long_metric_name', displayed_lines(updates)[-1])
            log_epoch({'train': {'loss': .35}, 'val': {'loss': .2}}, 1, 2, 1)
        text = stream.getvalue()
        self.assertEqual(text.count('\n'), 1)
        lines = displayed_lines(text)
        self.assertEqual(len(lines), 1)
        self.assertIn('E 1/2  TRAIN> loss: 0.3500', lines[0])
        self.assertIn('VAL> loss: 0.2000', lines[0])
        self.assertNotIn(' B ', lines[0])
        self.assertNotIn('a_long_metric_name', lines[0])

    def test_stage_colors_and_nonfinite_values_leave_normal_metrics_uncolored(self):
        with self.output(terminal=True) as stream:
            log_batch({'loss': .4}, 1, 1, 1, 1, 'TRAIN', 1, 1)
            log_batch({'loss': .3}, 1, 1, 1, 1, 'VAL', 1, 1)
            log_batch({'loss': float('nan')}, 1, 1, 1, 1, 'TEST', 1, 1)
            log_epoch({'train': {'loss': .4}, 'val': {'loss': .3}}, 1, 1, 1,
                      extra_info=_color('[3.65s]', 'duration'))
        text = stream.getvalue()
        self.assertIn('\x1b[36mTRAIN\x1b[0m', text)
        self.assertIn('\x1b[33mVAL\x1b[0m', text)
        self.assertIn('\x1b[35mTEST\x1b[0m', text)
        self.assertIn('\x1b[31mnan', text)
        self.assertIn('loss: 0.4000', text)
        self.assertNotIn('\x1b[31m0.4000', text)
        self.assertTrue(text.endswith('0.3000   \x1b[90m[3.65s]\x1b[0m\n'))

    def test_no_color_and_dumb_terminal_keep_batch_refresh_without_styles(self):
        for settings in ({'no_color': True}, {'term': 'dumb'}):
            with self.subTest(settings=settings), self.output(terminal=True, **settings) as stream:
                log_batch({'loss': .4}, 1, 1, 1, 1, 'TRAIN', 1, 1)
                self.assertEqual(_color('saved', 'success'), 'saved')
                log_epoch({'train': {'loss': .4}}, 1, 1, 1,
                          extra_info=_color('[3.65s]', 'duration'))
            self.assertIn('\r', stream.getvalue())
            self.assertNotRegex(stream.getvalue(), r'\x1b\[[0-9;]*m')
            self.assertTrue(stream.getvalue().endswith('0.4000   [3.65s]\n'))
            _clear_progress_line()

    def test_empty_metrics_still_show_batch_counts(self):
        with self.output() as stream:
            log_batch({}, 1, 1, 1, 3, 'TEST', 1, 1)
            log_batch({}, 1, 1, 2, 3, 'TEST', 1, 1)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('B 1/3  TEST>', lines[0])
        self.assertIn('B 2/3  TEST>', lines[1])

    def test_tqdm_keeps_each_epoch_summary_in_terminal_history(self):
        with self.output(terminal=True) as stream:
            with tqdm(total=2, file=stream, mininterval=0, bar_format='{desc}') as bar:
                for epoch in (1, 2):
                    log_batch({'loss': 1. / epoch}, epoch, 2, 1, 1, 'TRAIN', 1, 1, tqdm_iter=bar)
                    log_epoch({'train': {'loss': 1. / epoch}}, epoch, 2, 1, tqdm_iter=bar)
                    bar.update(1)
        lines = displayed_lines(stream.getvalue())
        summaries = [line for line in lines if '  TRAIN>' in line and ' B ' not in line]
        self.assertEqual(len(summaries), 2, lines)
        self.assertIn('E 1/2  TRAIN> loss: 1.0000', summaries[0])
        self.assertIn('E 2/2  TRAIN> loss: 0.5000', summaries[1])

    def test_disabled_tqdm_preserves_plain_redirected_batch_and_epoch_records(self):
        with self.output() as stream:
            with tqdm(total=1, file=stream, disable=True) as bar:
                log_batch({'loss': 1.}, 1, 1, 1, 1, 'TRAIN', 1, 1, tqdm_iter=bar)
                log_epoch({'train': {'loss': 1.}}, 1, 1, 1, tqdm_iter=bar)
        text = stream.getvalue()
        self.assertEqual(len(text.splitlines()), 2)
        self.assertNotIn('\r', text)
        self.assertNotIn('\x1b', text)

    def test_an_event_can_clear_the_active_batch_without_losing_the_next_summary(self):
        with self.output(terminal=True) as stream:
            log_batch({'a_long_metric_name': .5}, 1, 1, 1, 1, 'TRAIN', 1, 1)
            _clear_progress_line()
            print('Checkpoint saved')
            log_epoch({'train': {'loss': .5}}, 1, 1, 1)
        lines = displayed_lines(stream.getvalue())
        self.assertEqual(lines[0], 'Checkpoint saved')
        self.assertIn('TRAIN> loss: 0.5000', lines[1])
        self.assertNotIn('a_long_metric_name', '\n'.join(lines))

    def test_wrapped_batch_updates_replace_all_physical_rows(self):
        for no_color in (False, True):
            with self.subTest(no_color=no_color), self.output(terminal=True, no_color=no_color) as stream:
                print('Permanent header')
                first, second = 'first:' + 'x' * 107, 'second:' + 'y' * 154
                print_out(_color(first, 'train'), end='')
                print_out(_color(second, 'val'), end='')
                print_out(_color('latest: loss=0.25', 'train'), end='')
            text = stream.getvalue()
            self.assertIn(first, text)
            self.assertIn(second, text)
            self.assertEqual(displayed_lines(text), ['Permanent header', 'latest: loss=0.25'])

    def test_wrapped_progress_is_cleared_before_messages_and_epoch_summary(self):
        with self.output(terminal=True) as stream:
            print('Permanent header')
            print_out('old batch: ' + 'x' * 150, end='')
            _clear_progress_line()
            print('Checkpoint saved')
            print_out('next batch: ' + 'y' * 150, end='')
            log_epoch({'train': {'loss': .25}, 'val': {'loss': .2}}, 1, 2, 1)
        lines = displayed_lines(stream.getvalue())
        self.assertEqual(lines[:2], ['Permanent header', 'Checkpoint saved'])
        self.assertEqual(len(lines), 3, lines)
        self.assertIn('E 1/2  TRAIN> loss: 0.2500', lines[-1])
        self.assertIn('VAL> loss: 0.2000', lines[-1])
        self.assertNotIn('batch:', '\n'.join(lines))

    def test_exact_column_boundaries_preserve_the_previous_summary(self):
        for width in (79, 80, 81, 160, 161):
            with self.subTest(width=width), self.output(terminal=True) as stream:
                print('Previous epoch summary')
                print_out(_color('x' * width, 'train'), end='')
                print_out('next batch', end='')
            self.assertEqual(displayed_lines(stream.getvalue()), ['Previous epoch summary', 'next batch'])

    def test_progress_uses_stdout_columns_instead_of_environment_columns(self):
        with self.output(terminal=True) as stream, patch.dict(os.environ, {'COLUMNS': '200'}), \
                patch('deepepochs.loops.os.get_terminal_size', return_value=os.terminal_size((80, 24))) as size:
            print_out('x' * 113, end='')
            print_out('next batch', end='')
            self.assertEqual(size.call_args_list, [unittest.mock.call(stream.fileno()),
                                                 unittest.mock.call(stream.fileno())])
        self.assertEqual(displayed_lines(stream.getvalue()), ['next batch'])

    def test_zero_columns_and_size_errors_use_complete_plain_lines(self):
        for unavailable in (os.terminal_size((0, 24)), OSError('terminal size unavailable')):
            with self.subTest(unavailable=unavailable), self.output(terminal=True) as stream:
                query = {'side_effect': unavailable} if isinstance(unavailable, OSError) else {'return_value': unavailable}
                with patch('deepepochs.loops.os.get_terminal_size', **query):
                    log_batch({'main/loss': .5, 'main/accuracy': .75, 'main/recall': .625},
                              1, 2, 1, 2, 'TRAIN', 1, 1)
                    log_batch({'main/loss': .4, 'main/accuracy': .8, 'main/recall': .7},
                              1, 2, 2, 2, 'TRAIN', 1, 1)
                    log_epoch({'train': {'main/loss': .45}}, 1, 2, 1)
            raw = stream.getvalue()
            self.assertNotIn('\r', raw)
            self.assertNotIn('\x1b', raw)
            lines = raw.splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn('main/recall: 0.6250', lines[0])
            self.assertIn('main/recall: 0.7000', lines[1])
            self.assertIn('TRAIN> main/loss: 0.4500', lines[2])

    def test_tqdm_trainer_does_not_create_a_bar_when_size_query_fails(self):
        import torch
        from deepepochs import Trainer
        from torch.utils.data import DataLoader, TensorDataset

        network = torch.nn.Linear(1, 1)
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1)
        with self.output(terminal=True) as stream, \
                patch('deepepochs.loops.os.get_terminal_size', side_effect=OSError('unavailable')), \
                patch('deepepochs.trainer.trainer._ConsoleProgress') as bar:
            with Trainer(network, torch.nn.functional.mse_loss,
                         torch.optim.SGD(network.parameters(), lr=.1), epochs=1,
                         device='cpu', show_info=False, log_tqdm=True) as trainer:
                trainer.fit(loader)
            bar.assert_not_called()
        self.assertEqual(len(stream.getvalue().splitlines()), 3)
        self.assertNotIn('\r', stream.getvalue())
        self.assertNotIn('\x1b', stream.getvalue())

    def test_tqdm_wide_batches_stay_complete_through_refresh_events_and_summary(self):
        first = {'main/loss': .5, 'main/accuracy': .75, 'main/recall': .625}
        second = {'main/loss': .4, 'main/accuracy': .8, 'main/recall': .7}
        for no_color in (False, True):
            with self.subTest(no_color=no_color), self.output(terminal=True, no_color=no_color) as stream:
                print('Permanent header')
                with _ConsoleProgress(total=2, file=stream, ncols=80, mininterval=0, leave=False,
                                      bar_format='{percentage:3.0f}%|{bar}| {desc}') as bar:
                    log_batch(first, 1, 2, 1, 2, 'TRAIN', 1, 1, tqdm_iter=bar)
                    for values in (first,):
                        visible = ''.join(displayed_lines(stream.getvalue())).replace(' ', '')
                        for key, value in values.items():
                            self.assertIn(f'{key}:{value:.4f}', visible)
                    bar.update(1)
                    refreshed = ''.join(displayed_lines(stream.getvalue())).replace(' ', '')
                    self.assertEqual(refreshed.count('E1/2B1/2'), 1)
                    self.assertIn('main/recall:0.6250', refreshed)
                    with tqdm.external_write_mode(file=stream):
                        print('Ordinary event')
                    log_batch(second, 1, 2, 2, 2, 'TRAIN', 1, 1, tqdm_iter=bar)
                    visible = ''.join(displayed_lines(stream.getvalue())).replace(' ', '')
                    self.assertIn('Ordinaryevent', visible)
                    self.assertEqual(visible.count('E1/2B2/2'), 1)
                    self.assertNotIn('E1/2B1/2', visible)
                    for key, value in second.items():
                        self.assertIn(f'{key}:{value:.4f}', visible)
                    log_epoch({'train': second}, 1, 2, 1, tqdm_iter=bar)
                    bar.update(1)
            final = ''.join(displayed_lines(stream.getvalue())).replace(' ', '')
            self.assertIn('PermanentheaderOrdinaryevent', final)
            self.assertEqual(final.count('E1/2TRAIN>'), 1)
            self.assertNotIn('B1/2', final)
            self.assertNotIn('B2/2', final)
            self.assertNotIn('|', final)
            for key, value in second.items():
                self.assertIn(f'{key}:{value:.4f}', final)

    def test_tqdm_close_clears_a_wrapped_uncommitted_batch(self):
        with self.output(terminal=True) as stream:
            print('Permanent header')
            with _ConsoleProgress(total=2, file=stream, ncols=80, mininterval=0, leave=False,
                                  bar_format='{percentage:3.0f}%|{bar}| {desc}') as bar:
                log_batch({'main/loss': .5, 'main/accuracy': .75, 'main/recall': .625},
                          1, 2, 1, 2, 'TRAIN', 1, 1, tqdm_iter=bar)
                bar.update(1)
            print('Stopped')
        self.assertEqual(displayed_lines(stream.getvalue()), ['Permanent header', 'Stopped'])

    def test_small_nonzero_values_and_all_learning_rate_groups_remain_visible(self):
        optimizers = [SimpleNamespace(param_groups=[{'lr': 2e-4}, {'lr': 3e-6}]),
                      SimpleNamespace(param_groups=[{'lr': 1e-5}])]
        with self.output():
            short = info({'loss': 1e-8, 'accuracy': .123456789}, 4)
            precise = info({'loss': -1e-10, 'accuracy': .123456789}, 7)
            rates = _format_learning_rates(optimizers)
            single_rate = _format_learning_rates(optimizers[1:])
        self.assertIn('loss: 1.000e-08', short)
        self.assertIn('accuracy: 0.1235', short)
        self.assertIn('loss: -1.000000e-10', precise)
        self.assertIn('accuracy: 0.1234568', precise)
        self.assertEqual(rates, 'opt0/group0=2.000e-04 opt0/group1=3.000e-06 opt1/group0=1.000e-05')
        self.assertEqual(single_rate, '1.000e-05')


if __name__ == '__main__':
    unittest.main()
