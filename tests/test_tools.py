import inspect
import os
import shlex
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from deepepochs.callbacks.log import run_tensorboard
from deepepochs.tools import load_yaml


class YamlTests(unittest.TestCase):
    def load(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(content, encoding='utf-8')
            return load_yaml(path)

    def test_documented_yaml_example(self):
        example = inspect.getdoc(load_yaml).split('yaml file content:\n', 1)[1].split('load result:', 1)[0]
        result = self.load(textwrap.dedent(example))
        self.assertEqual(result, {
            'data_root': './data',
            'path': os.path.join('./data', 'filename'),
            'str': './datafilename',
            'items': [1, 2, 3],
            'key': {'key1': 'value1', 'key2': 'value2'},
        })

    def test_current_arithmetic_tags(self):
        result = self.load('echo: !echo [hello]\nsum: !sum [1, 2, 3]\nmul: !mul [2, 3, 4]\n')
        self.assertEqual(result, {'echo': 'hello', 'sum': 6, 'mul': 24})

@unittest.skipUnless(os.name == 'posix', 'requires a POSIX executable fixture')
class TensorBoardTests(unittest.TestCase):
    def test_tensorboard_paths_with_spaces_are_passed_as_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'python environment'
            binary = prefix / 'bin/tensorboard'
            binary.parent.mkdir(parents=True)
            arguments = Path(directory) / 'arguments.txt'
            binary.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > '
                              + shlex.quote(str(arguments)) + '\nexit 7\n')
            binary.chmod(0o755)
            log_dir = Path(directory) / 'event logs'
            with patch('deepepochs.callbacks.log.sys.prefix', str(prefix)), \
                    patch('deepepochs.callbacks.log.platform.system', return_value='Linux'):
                self.assertIsNone(run_tensorboard(log_dir))
            self.assertEqual(arguments.read_text().splitlines(), ['--logdir', str(log_dir)])


if __name__ == '__main__':
    unittest.main()
