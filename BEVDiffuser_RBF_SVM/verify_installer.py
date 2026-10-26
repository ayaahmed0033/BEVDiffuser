#!/usr/bin/env python3
"""Standard-library installer checks; never touches the user's repository."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import json
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import install_shape_svm as installer

PACKAGE = Path(__file__).resolve().parent
VARIANTS = ('baseline', 'raf', 'evssm_baseline', 'evssm_raf')


def snapshot(repo):
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in repo.rglob('*') if p.is_file() and '.git' not in p.relative_to(repo).parts}


def fixture(parent, variant):
    repo = parent / variant
    target = repo / installer.REL
    target.mkdir(parents=True)
    for path in (PACKAGE / 'baseline_checks' / variant).iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    for name in ('train.sh', 'test.sh'):
        text = (target / name).read_text()
        text = text.replace('29503', '33333').replace('9999', '33334')
        text = text.replace('/home/aya/BEVDiffuser/BEVFormer/hf_models/stable-diffusion-2-1', '/custom/hf')
        (target / name).write_text(text)
    train = target / 'train.sh'
    text = train.read_text().replace('    "$@"', '    --use_raf \\\n    "$@"')
    if '--use_raf' not in text:
        text += '\n# existing custom settings\n'
    train.write_text(text)
    for name in ('train_bev_diffuser.py', 'test_bev_diffuser.py'):
        path = target / name
        path.write_text(path.read_text().replace('/home/aya/BEVDiffuser/BEVFormer/hf_models/stable-diffusion-2-1', '/custom/hf')
                        + '\n# USER_ENV_SENTINEL\n')
    for name in ('BEVFormer/tools/create_data.py', 'BEVFormer/data/sensor_sentinel',
                 'BEVFormer/projects/bevdiffuser/layout_diffusion/layout_diffusion_unet.py'):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# ORIGINAL_UNTOUCHED\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
    return repo


def invoke(repo, *args):
    return subprocess.run([sys.executable, str(PACKAGE / 'install_shape_svm.py'),
                           '--repo', str(repo)] + list(args), text=True, capture_output=True)


class InstallerTests(unittest.TestCase):
    def test_apply_preserves_settings_creates_backup_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    repo = fixture(parent, variant)
                    before = snapshot(repo)
                    preview = invoke(repo)
                    self.assertEqual(preview.returncode, 0, preview.stderr)
                    self.assertEqual(snapshot(repo), before)
                    result = invoke(repo, '--apply')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('Backup:', result.stdout)
                    for name in ('train_bev_diffuser.py', 'test_bev_diffuser.py'):
                        text = (repo / installer.REL / name).read_text()
                        self.assertIn('USER_ENV_SENTINEL', text)
                        self.assertIn('/custom/hf', text)
                        if variant.startswith('evssm'):
                            self.assertIn('configure_evssm', text)
                    train = (repo / installer.REL / 'train.sh').read_text()
                    test = (repo / installer.REL / 'test.sh').read_text()
                    self.assertIn('33333', train)
                    self.assertIn('33334', test)
                    for relative in before:
                        if relative not in [str(installer.REL / name) for name in
                            ('train_bev_diffuser.py', 'test_bev_diffuser.py', 'model_utils.py', 'train.sh', 'test.sh')]:
                            self.assertEqual(snapshot(repo)[relative], before[relative])
                    after = snapshot(repo)
                    repeat = invoke(repo, '--apply')
                    self.assertEqual(repeat.returncode, 0, repeat.stderr)
                    self.assertIn('already installed', repeat.stdout)
                    self.assertEqual(snapshot(repo), after)

    def test_both_v2_helpers_are_safe_upgrade_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = fixture(Path(directory), 'baseline')
            result = invoke(repo, '--apply')
            self.assertEqual(result.returncode, 0, result.stderr)
            target = repo / installer.REL / 'shape_integration.py'
            previous = (PACKAGE / 'baseline_checks' / 'previous_shape_integration.py').read_bytes()
            old_line = b'        counts += np.bincount(selected, minlength=10)\n'
            fixed_lines = (
                b'        # Empty annotation lists can become float64 arrays in nuScenes.\n'
                b'        # bincount requires integer IDs even when its input is empty.\n'
                b'        if selected.size:\n'
                b'            if not np.all(selected == np.floor(selected)):\n'
                b"                raise ValueError('Training class labels must be integer IDs')\n"
                b'            counts += np.bincount(selected.astype(np.int64), minlength=10)\n')
            previous_with_empty_label_fix = previous.replace(old_line, fixed_lines)
            self.assertEqual(previous.count(old_line), 1)
            self.assertEqual(hashlib.sha256(previous_with_empty_label_fix).hexdigest(),
                             '5e3bd4de2becb04447b9dc7c4e37a786603d55ba57eca3c0d65112bfc79b6aa4')
            current = (PACKAGE / 'files' / installer.REL / 'shape_integration.py').read_bytes()
            for known_previous in (previous, previous_with_empty_label_fix):
                target.write_bytes(known_previous)
                result = invoke(repo, '--apply')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(target.read_bytes(), current)

                target.write_bytes(known_previous + b'\n# USER_CUSTOM_CHANGE\n')
                before = snapshot(repo)
                result = invoke(repo, '--apply')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Nothing installed', result.stderr)
                self.assertEqual(snapshot(repo), before)

    def test_known_v2_experiment_tests_are_safe_upgrade_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = fixture(Path(directory), 'baseline')
            result = invoke(repo, '--apply')
            self.assertEqual(result.returncode, 0, result.stderr)
            target = repo / 'tests' / 'test_shape_experiment.py'
            previous = (PACKAGE / 'baseline_checks' / 'previous_test_shape_experiment.py').read_bytes()
            self.assertEqual(hashlib.sha256(previous).hexdigest(),
                             '9c76ee2426db85efcfbf60dfa0bc2d8206f082bf09a91d4f68801963bde83599')
            target.write_bytes(previous)
            result = invoke(repo, '--apply')
            self.assertEqual(result.returncode, 0, result.stderr)
            current = (PACKAGE / 'files' / 'tests' / 'test_shape_experiment.py').read_bytes()
            self.assertEqual(target.read_bytes(), current)

            target.write_bytes(previous + b'\n# USER_CUSTOM_CHANGE\n')
            before = snapshot(repo)
            result = invoke(repo, '--apply')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Nothing installed', result.stderr)
            self.assertEqual(snapshot(repo), before)

    def test_shell_forwards_extra_options_and_consumes_gpu_count(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    repo = fixture(parent, variant)
                    result = invoke(repo, '--apply')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    binary = parent / ('bin_' + variant)
                    binary.mkdir()
                    fake_python = binary / 'python'
                    fake_python.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                                           'open(os.environ["SHAPE_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
                    fake_python.chmod(0o755)
                    capture = parent / ('args_' + variant + '.json')
                    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], SHAPE_CAPTURE=str(capture))
                    flags = ['--use_svm', '--svm_kernel', 'rbf', '--svm_rbf_features', '256',
                             '--grayscale_aug', '--use_unet_bottleneck',
                             '--bottleneck_time_gate', '--svm_class_balance']
                    cwd = repo / installer.REL
                    for script, extra in [('train.sh', flags + ['--max_train_steps', '2']), ('test.sh', ['--eval', 'bbox'])]:
                        process = subprocess.run(['bash', script, '1'] + extra, cwd=cwd, env=env,
                                                 capture_output=True, text=True)
                        self.assertEqual(process.returncode, 0, process.stderr)
                        args = json.loads(capture.read_text())
                        self.assertEqual(args[-len(extra):], extra)
                        for i, value in enumerate(args):
                            if value == '1':
                                self.assertGreater(i, 0)
                                self.assertEqual(args[i-1], '--gradient_accumulation_steps')
                        self.assertNotIn('--use_raf', args)
                        self.assertNotIn('--no_raf', args)
                        for flag in extra:
                            self.assertIn(flag, args)
                        self.assertIn('--nproc_per_node=1', args)

    def test_patched_parsers_accept_all_selected_flags(self):
        helper = ast.parse((PACKAGE / 'files' / installer.REL / 'shape_integration.py').read_text())
        add_args = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'add_shape_arguments')
        with tempfile.TemporaryDirectory() as directory:
            for variant in VARIANTS:
                repo = fixture(Path(directory), variant)
                result = invoke(repo, '--apply')
                self.assertEqual(result.returncode, 0, result.stderr)
                for name in ('train_bev_diffuser.py', 'test_bev_diffuser.py'):
                    tree = ast.parse((repo / installer.REL / name).read_text())
                    parse_args = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
                    namespace = dict(argparse=argparse, os=os, DictAction=argparse.Action,
                                     add_evssm_args=lambda parser: None)
                    module = ast.Module(body=[add_args, parse_args], type_ignores=[])
                    exec(compile(module, name, 'exec'), namespace)
                    flags = ['--use_svm', '--svm_kernel', 'rbf', '--svm_rbf_features', '256',
                             '--grayscale_aug', '--use_unet_bottleneck',
                             '--bottleneck_time_gate', '--svm_class_balance']
                    if name.startswith('train'):
                        flags += ['--max_train_steps', '2', '--skip_checkpoint_eval']
                    with mock.patch.object(sys, 'argv', ['script'] + flags):
                        args = namespace['parse_args']()
                    self.assertTrue(args.use_svm and args.bottleneck_time_gate and args.svm_class_balance)
                    self.assertEqual(args.svm_kernel, 'rbf')
                    self.assertEqual(args.svm_rbf_features, 256)
                    if name.startswith('train'):
                        self.assertEqual(args.max_train_steps, 2)

    def test_incompatible_anchor_installs_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = fixture(Path(directory), 'baseline')
            path = repo / installer.REL / 'model_utils.py'
            path.write_text(path.read_text().replace('    cfg.model.pretrained = None', '    cfg.model.pretrained = "CUSTOM"'))
            before = snapshot(repo)
            result = invoke(repo, '--apply')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Nothing installed', result.stderr)
            self.assertEqual(snapshot(repo), before)

    def test_active_merge_installs_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = fixture(Path(directory), 'baseline')
            (repo / '.git' / 'MERGE_HEAD').write_text('conflict')
            before = snapshot(repo)
            result = invoke(repo, '--apply')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('active Git operation', result.stderr)
            self.assertEqual(snapshot(repo), before)

    def test_write_failure_restores_previous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = fixture(Path(directory), 'baseline')
            before = snapshot(repo)
            original = Path.write_bytes
            attempts = [0]
            def fail_third(path, data):
                if repo in path.parents:
                    attempts[0] += 1
                    if attempts[0] == 3:
                        raise RuntimeError('simulated write failure')
                return original(path, data)
            with mock.patch.object(sys, 'argv', ['install_shape_svm.py', '--repo', str(repo), '--apply']):
                with mock.patch.object(Path, 'write_bytes', fail_third):
                    with self.assertRaisesRegex(RuntimeError, 'simulated write failure'):
                        with redirect_stdout(io.StringIO()):
                            installer.main()
            self.assertEqual(snapshot(repo), before)


if __name__ == '__main__':
    unittest.main()
