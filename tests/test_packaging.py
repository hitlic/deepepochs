import ast
import email
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("build") is None:
            raise unittest.SkipTest("Packaging tests require the build package")
        cls.temp = tempfile.TemporaryDirectory(prefix="deepepochs-packaging-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        snapshot = cls.base / "source"
        snapshot.mkdir()
        for filename in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(PROJECT_ROOT / filename, snapshot / filename)
        shutil.copytree(
            SOURCE_ROOT / "deepepochs",
            snapshot / "src" / "deepepochs",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        cls.source_files = {
            "deepepochs/" + path.relative_to(SOURCE_ROOT / "deepepochs").as_posix()
            for path in (SOURCE_ROOT / "deepepochs").rglob("*.py")
        }
        module = ast.parse((SOURCE_ROOT / "deepepochs" / "__init__.py").read_text())
        cls.version = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        )
        cls.dist = cls.base / "dist"
        result = subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", str(snapshot), "--outdir", str(cls.dist)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode:
            raise AssertionError(result.stdout[-8000:])
        cls.wheel = next(cls.dist.glob("*.whl"))

    def test_sdist_and_wheel_include_current_source(self):
        with zipfile.ZipFile(self.wheel) as archive:
            wheel_files = set(archive.namelist())
            self.assertTrue(self.source_files.issubset(wheel_files))
            self.assertFalse(any(name.startswith("src/") for name in wheel_files))
        with tarfile.open(next(self.dist.glob("*.tar.gz"))) as archive:
            names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
            self.assertTrue({"pyproject.toml", "README.md", "LICENSE"}.issubset(names))
            self.assertTrue({"src/" + name for name in self.source_files}.issubset(names))

    def test_metadata_matches_runtime_version_and_dependencies(self):
        from packaging.requirements import Requirement

        with zipfile.ZipFile(self.wheel) as archive:
            name = next(name for name in archive.namelist() if name.endswith("/METADATA"))
            metadata = email.message_from_bytes(archive.read(name))
        self.assertEqual(metadata["Name"], "deepepochs")
        self.assertEqual(metadata["Version"], self.version)
        self.assertEqual(metadata["Requires-Python"], ">=3.8")
        requirements = {
            requirement.name: requirement
            for requirement in map(Requirement, metadata.get_all("Requires-Dist", []))
        }
        self.assertEqual(str(requirements["torch"].specifier), ">=2.0.0")
        self.assertNotIn("accelerate", requirements)

    def test_wheel_imports_without_accelerate(self):
        installed = self.base / "installed"
        with zipfile.ZipFile(self.wheel) as archive:
            archive.extractall(installed)
        script = """
import json
import sys
sys.modules['accelerate'] = None
import deepepochs
from deepepochs import Trainer
print(json.dumps({'version': deepepochs.__version__, 'path': deepepochs.__file__, 'trainer': Trainer.__name__}))
"""
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(installed),
            PYTHONDONTWRITEBYTECODE="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(self.base / "matplotlib"),
            XDG_CACHE_HOME=str(self.base / "cache"),
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.base,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        imported = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(imported["version"], self.version)
        self.assertEqual(imported["trainer"], "Trainer")
        self.assertIn(installed, Path(imported["path"]).parents)


if __name__ == "__main__":
    unittest.main()
