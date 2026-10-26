#!/usr/bin/env python3
"""Install/upgrade opt-in kernel SVM, grayscale and U-Net adapters safely.

python install_shape_svm.py --repo /home/aya/BEVDiffuser --apply
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PACKAGE = Path(__file__).resolve().parent
REL = Path('BEVFormer/projects/bevdiffuser')
MARKER = '# SHAPE SVM optional integration v3'
KNOWN_MARKERS = (MARKER, '# SHAPE SVM optional integration v2')
IMPORT_TRAIN = '''from projects.bevdiffuser.shape_integration import (
    add_shape_arguments, active as shape_active, restore_options, validate as validate_shape,
    configure_training_data, configure_unet, load_shape_unet, register_shape_metadata,
    configure_class_balance)'''
IMPORT_TEST = '''from projects.bevdiffuser.shape_integration import (
    add_shape_arguments, active as shape_active, restore_options,
    validate as validate_shape, configure_unet, load_shape_unet)'''
TRAIN_SETUP = '''    shape_checkpoint = args.resume_from_checkpoint
    if shape_checkpoint:
        restore_options(args, shape_checkpoint, exact=True)
    elif args.pretrained_unet_checkpoint and os.path.isdir(args.pretrained_unet_checkpoint):
        restore_options(args, args.pretrained_unet_checkpoint)
    validate_shape(args, training=True)
    args._shape_exact = bool(shape_checkpoint)
    if shape_active(args):
        if not args.output_dir.rstrip('/').endswith('-shape'):
            args.output_dir = args.output_dir.rstrip('/') + '-shape'
        args.tracker_run_name = (args.tracker_run_name or 'BEVDiffuser') + '-shape'
    if args.use_svm and shape_checkpoint:
        args.bev_checkpoint = os.path.join(shape_checkpoint, 'bev_model.pth')
    elif args.use_svm and args.pretrained_unet_checkpoint and os.path.isdir(args.pretrained_unet_checkpoint):
        if os.path.isfile(os.path.join(args.pretrained_unet_checkpoint, 'shape_options.json')):
            args.bev_checkpoint = os.path.join(args.pretrained_unet_checkpoint, 'bev_model.pth')
            args._shape_exact = True
'''
TEST_SETUP = '''    restore_options(args, args.checkpoint_dir, exact=True)
    validate_shape(args)
    args._shape_exact = True
    if args.use_svm:
        args.bev_checkpoint = os.path.join(args.checkpoint_dir, 'bev_model.pth')
'''


def replace_once(text, needle, replacement):
    if text.count(needle) != 1:
        raise ValueError('Expected one integration anchor: ' + needle.strip()[:100])
    return text.replace(needle, replacement, 1)


def patch_python(text, name):
    if any(marker in text for marker in KNOWN_MARKERS):
        return text
    if 'projects.bevdiffuser.shape_integration' in text:
        raise ValueError('Unmarked earlier shape integration found; compare before installing')
    if name == 'model_utils.py':
        needle = '    cfg.model.pretrained = None'
        text = replace_once(text, needle, needle + '''
    from projects.bevdiffuser.shape_integration import configure_model
    configure_model(cfg, args)''')
        needle = "    checkpoint = load_checkpoint(model, args.bev_checkpoint, map_location='cpu')"
        text = replace_once(text, needle, needle + '''
    if getattr(args, '_shape_exact', False) and args.use_svm:
        state = checkpoint.get('state_dict', checkpoint)
        keys = {key[7:] if key.startswith('module.') else key for key in state}
        expected = {key for key in model.state_dict() if '.shape_svm.' in key}
        if not expected.issubset(keys):
            raise ValueError('Evaluation/resume requires bev_model.pth containing all trained SVM weights')''')
    else:
        train = name == 'train_bev_diffuser.py'
        needle = 'from test_bev_diffuser import evaluate' if train else 'from model_utils import get_bev_model, build_unet'
        text = replace_once(text, needle, needle + '\n' + (IMPORT_TRAIN if train else IMPORT_TEST))
        parser = '    parser = argparse.ArgumentParser(description="Simple example of a training script.")'
        text = replace_once(text, parser, parser + '\n    add_shape_arguments(parser)')
        setup_anchor = ('def train():' if train else 'def test():') + '\n    args = parse_args()'
        text = replace_once(text, setup_anchor, setup_anchor + '\n' + (TRAIN_SETUP if train else TEST_SETUP).rstrip())
        if train:
            needle = '        bev_cfg.merge_from_dict(args.cfg_options)'
            text = replace_once(text, needle, needle + '\n    configure_training_data(bev_cfg, args)')
            needle = '        bev_model.module.pts_bbox_head.reg_branches.requires_grad_(True)'
            text = replace_once(text, needle, needle + '''
    if args.use_svm:
        bev_model.module.pts_bbox_head.shape_svm.requires_grad_(True)''')
            needle = '    unet = build_unet(bev_cfg.unet)'
            text = replace_once(text, needle, needle + '''
    adapter_count = configure_unet(unet, args)
    if adapter_count:
        logger.info('Installed %s residual U-Net bottlenecks', adapter_count)''')
            if "        if args.use_raf:\n            strict_load_unet(unet, args.pretrained_unet_checkpoint)" in text:
                needle = "        if args.use_raf:\n            strict_load_unet(unet, args.pretrained_unet_checkpoint)"
                text = replace_once(text, needle, '''        if shape_active(args):
            load_shape_unet(unet, args.pretrained_unet_checkpoint, allow_baseline=True)
        elif args.use_raf:
            strict_load_unet(unet, args.pretrained_unet_checkpoint)''')
            else:
                needle = '        unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder="unet")'
                text = replace_once(text, needle, '''        if shape_active(args):
            load_shape_unet(unet, args.pretrained_unet_checkpoint, allow_baseline=True)
        else:
            unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder="unet")''')
            needle = '        unet.upsample_blocks.requires_grad_(True)'
            text = replace_once(text, needle, needle + '''
        if args.use_unet_bottleneck:
            unet.shape_adapters.requires_grad_(True)''')
            needle = '            model.from_pretrained(os.path.join(input_dir, "unet"))'
            text = replace_once(text, needle, '''            if shape_active(args):
                load_shape_unet(model, input_dir)
            else:
                model.from_pretrained(os.path.join(input_dir, "unet"))''')
            needle = '    if args.gradient_checkpointing:'
            text = replace_once(text, needle, '    register_shape_metadata(accelerator, args)\n\n' + needle)
            needle = '    train_dataloader = build_dataloader('
            text = replace_once(text, needle, '''    class_counts = configure_class_balance(bev_model, train_dataset, args, bev_cfg.point_cloud_range)
    if class_counts is not None:
        logger.info('Class-balanced SVM training counts: %s', class_counts)
''' + needle)
            if '--skip_checkpoint_eval' not in text:
                text = replace_once(text, '    add_shape_arguments(parser)', '''    add_shape_arguments(parser)
    parser.add_argument('--skip_checkpoint_eval', action='store_true', help='Skip checkpoint validation for a short smoke run')''')
                start = text.index('                    unet.eval()')
                ending = text.index('                    unet.train()', start)
                ending = text.find('\n', ending)
                if ending < 0:
                    raise ValueError('Cannot locate the checkpoint evaluation block')
                block = text[start:ending]
                text = text[:start] + '                    if not args.skip_checkpoint_eval:\n' + '\n'.join(
                    '    ' + line if line.strip() else line for line in block.split('\n')) + text[ending:]
        else:
            needle = '    unet = build_unet(bev_cfg.unet)'
            text = replace_once(text, needle, needle + '\n    configure_unet(unet, args)')
            if "    if args.use_raf:\n        strict_load_unet(unet, args.checkpoint_dir)" in text:
                needle = "    if args.use_raf:\n        strict_load_unet(unet, args.checkpoint_dir)"
                text = replace_once(text, needle, '''    if shape_active(args):
        load_shape_unet(unet, args.checkpoint_dir)
    elif args.use_raf:
        strict_load_unet(unet, args.checkpoint_dir)''')
            else:
                needle = '    unet.from_pretrained(args.checkpoint_dir, subfolder="unet")'
                text = replace_once(text, needle, '''    if shape_active(args):
        load_shape_unet(unet, args.checkpoint_dir)
    else:
        unet.from_pretrained(args.checkpoint_dir, subfolder="unet")''')
    text = MARKER + '\n' + text
    ast.parse(text)
    return text


def patch_shell(text, train):
    if any(marker in text for marker in KNOWN_MARKERS):
        return text
    # RAF is selected explicitly at invocation; preserve all path/hyperparameter settings.
    text = re.sub(r'^[ \t]*---?use_raf[ \t]*\\?[ \t]*(?:\n|$)', '', text, flags=re.M)
    text = re.sub(r'^[ \t]*--no_raf[ \t]*\\?[ \t]*(?:\n|$)', '', text, flags=re.M)
    if train:
        if not re.search(r'^shift(?:\s|$)', text, re.M):
            if not re.search(r'^GPUS=', text, re.M):
                raise ValueError('Cannot locate training GPU argument')
            text = re.sub(r'^(GPUS=.*)$', r'\1\nshift', text, count=1, flags=re.M)
    elif '--nproc_per_node' not in text:
        prefix = '''SHAPE_TEST_GPUS=1
if [[ ${1:-} =~ ^[0-9]+$ ]]; then SHAPE_TEST_GPUS=$1; shift; fi
'''
        needle = 'python -m torch.distributed.launch '
        text = replace_once(text, needle, needle + '--nproc_per_node="$SHAPE_TEST_GPUS" ')
        text = prefix + text
    if '"$@"' not in text:
        lines = text.splitlines()
        active_options = [i for i, line in enumerate(lines) if line.lstrip().startswith('--')]
        if not active_options:
            raise ValueError('Cannot locate launcher arguments')
        last = active_options[-1]
        lines[last] = lines[last].rstrip().rstrip('\\').rstrip() + ' \\'
        lines.insert(last + 1, '    "$@"')
        text = '\n'.join(lines) + '\n'
    text = re.sub(r'^#!/[^\n]+\n', '', text)
    return '#!/usr/bin/env bash\n' + MARKER + '\n' + text


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / REL).is_dir():
        raise SystemExit('Expected BEVDiffuser repository with BEVFormer/projects/bevdiffuser/')
    git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=repo, text=True).strip()
    if Path(git_root).resolve() != repo:
        raise SystemExit('--repo must name the Git repository root')
    for state in ('rebase-merge', 'rebase-apply', 'MERGE_HEAD', 'CHERRY_PICK_HEAD'):
        found = Path(subprocess.check_output(['git', 'rev-parse', '--git-path', state], cwd=repo, text=True).strip())
        if (found if found.is_absolute() else repo / found).exists():
            raise SystemExit('Finish the active Git operation before installing: ' + state)
    if subprocess.check_output(['git', 'ls-files', '-u'], cwd=repo):
        raise SystemExit('Resolve unmerged files before installing')
    manifest = json.loads((PACKAGE / 'package_manifest.json').read_text())
    for item in manifest['package_files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise SystemExit('Invalid package path')
        source = PACKAGE / relative
        if source.is_symlink() or not source.is_file() or sha(source.read_bytes()) != item['sha256']:
            raise SystemExit('Package file checksum mismatch: ' + str(relative))
    writes = []
    try:
        for name in ('train_bev_diffuser.py', 'test_bev_diffuser.py', 'model_utils.py', 'train.sh', 'test.sh'):
            path = repo / REL / name
            if not path.is_file() or path.is_symlink() or repo not in path.resolve().parents:
                raise ValueError('Expected regular repository file: ' + str(path))
            text = path.read_text()
            new = patch_shell(text, name == 'train.sh') if name.endswith('.sh') else patch_python(text, name)
            writes.append((path, new.encode(), 0o755 if name.endswith('.sh') else path.stat().st_mode & 0o777))
        for item in manifest['install_files']:
            relative = Path(item)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid install path')
            destination = repo / relative
            source = PACKAGE / 'files' / relative
            if destination.is_symlink() or repo not in destination.resolve().parents:
                raise ValueError('External destination: ' + str(relative))
            if destination.exists() and (not destination.is_file() or destination.read_bytes() != source.read_bytes()):
                known_previous = manifest.get('upgrade_from', {}).get(relative.as_posix(), [])
                if isinstance(known_previous, str):
                    known_previous = [known_previous]
                if sha(destination.read_bytes()) not in known_previous:
                    raise ValueError('Existing shape file differs from known prior package; compare before replacing: '
                                     + str(relative))
            writes.append((destination, source.read_bytes(), 0o644))
        for path, content, _ in writes:
            if path.suffix == '.py':
                ast.parse(content.decode(), filename=str(path))
            elif path.suffix == '.sh':
                result = subprocess.run(['bash', '-n'], input=content, capture_output=True)
                if result.returncode:
                    raise ValueError('Invalid launcher syntax: ' + str(path))
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise SystemExit(str(error) + '\nNothing installed; your existing files are unchanged.')
    changed = [(p, data, mode) for p, data, mode in writes
               if not p.exists() or p.read_bytes() != data or p.stat().st_mode & 0o777 != mode]
    for path, _, _ in changed:
        print('INSTALL ' + str(path))
    if not args.apply:
        print('Preview only. Add --apply to back up and install.')
        return
    if not changed:
        print('Shape SVM integration is already installed.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = Path(tempfile.mkdtemp(prefix=repo.name + '_before_shape_svm_' + stamp + '_', dir=repo.parent))
    previous = []
    for path, _, _ in changed:
        relative = path.relative_to(repo)
        previous.append(dict(path=str(relative), existed=path.exists()))
        if path.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    (backup / 'backup_manifest.json').write_text(json.dumps(previous, indent=2) + '\n')
    attempted = []
    try:
        for path, data, mode in changed:
            attempted.append(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(mode)
            if sha(path.read_bytes()) != sha(data):
                raise RuntimeError('Installed file checksum mismatch: ' + str(path))
    except Exception:
        for path in reversed(attempted):
            saved = backup / path.relative_to(repo)
            if saved.exists():
                shutil.copy2(saved, path)
            elif path.is_file():
                path.unlink()
        raise
    print('Backup: ' + str(backup))
    print('Installed. Enable: bash train.sh 1 --use_svm --svm_kernel rbf --grayscale_aug --use_unet_bottleneck --bottleneck_time_gate --svm_class_balance')


if __name__ == '__main__':
    main()
