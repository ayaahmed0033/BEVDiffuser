# SPDX-License-Identifier: AGPL-3.0
"""Optional cached RGB EVSSM deblurring before BEVFormer image normalization.
EVSSM runs separately with its official CUDA dependencies; this module uses
only the existing MMDetection3D image loader when --use_evssm is enabled.
"""
import hashlib
import json
import os
import sqlite3
from pathlib import Path


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def add_evssm_args(parser):
    parser.add_argument('--use_evssm', action='store_true',
                        help='Use pretrained EVSSM camera-image cache in train and test')
    parser.add_argument('--evssm_cache', default=None, help='Root of prepared EVSSM PNG cache')
    parser.add_argument('--evssm_data_root', default=None, help='Original nuScenes image root')


class CacheIndex:
    def __init__(self, cache_root, data_root):
        self.root = Path(cache_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().absolute()
        with (self.root / 'manifest.json').open() as stream:
            self.manifest = json.load(stream)
        if self.manifest.get('version') != 1 or self.manifest.get('method') != 'EVSSM':
            raise ValueError('Not a supported EVSSM cache manifest')
        if self.manifest.get('status') != 'complete':
            raise ValueError('EVSSM cache is incomplete: finish prepare_evssm_cache.py first')
        self.identity = self.manifest['identity']
        self.image_count = self.manifest['image_count']
        if not self.image_count:
            raise ValueError('EVSSM cache contains no images')
        self.verified = set()
        self.connection = None
        self.pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['connection'] = None
        state['pid'] = None
        state['verified'] = set()
        return state

    def record(self, key):
        # Lazy per-process read-only SQLite connection: no GPU code in workers.
        if self.pid != os.getpid() or self.connection is None:
            if self.connection is not None:
                self.connection.close()
            self.connection = sqlite3.connect((self.root / 'index.sqlite').as_uri() + '?mode=ro', uri=True)
            self.pid = os.getpid()
            self.verified = set()
        row = self.connection.execute('SELECT record FROM images WHERE path=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def resolve(self, source):
        source_path = Path(source).expanduser()
        # Dataset filenames may be relative to CWD or to data_root.
        if not source_path.is_absolute():
            if source_path.is_file():
                source_path = source_path.resolve()
            else:
                source_path = self.data_root / source_path
        source_path = source_path.absolute()
        try:
            key = source_path.relative_to(self.data_root).as_posix()
        except ValueError:
            # Resolve links only as a fallback; preserve sample/sweep relative paths.
            try:
                key = source_path.resolve().relative_to(self.data_root.resolve()).as_posix()
            except ValueError:
                raise ValueError('Camera image is outside --evssm_data_root: ' + str(source))
        record = self.record(key)
        if record is None:
            raise FileNotFoundError('No EVSSM cache for ' + key + '; prepare all camera samples and sweeps')
        output = (self.root / record['output']).resolve()
        if self.root not in output.parents:
            raise ValueError('Invalid cached-image path')
        if key not in self.verified:
            if sha256(source_path) != record['source_sha256']:
                raise ValueError('Original camera image changed since EVSSM preprocessing: ' + key)
            if sha256(output) != record['output_sha256']:
                raise ValueError('EVSSM cached image changed: ' + key)
            self.verified.add(key)
        return str(output), record


def register_loader():
    from mmdet.datasets.builder import PIPELINES
    from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles
    if PIPELINES.get('LoadEVSSMMultiViewImageFromFiles') is not None:
        return

    @PIPELINES.register_module()
    class LoadEVSSMMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
        def __init__(self, evssm_cache, evssm_data_root, **kwargs):
            super().__init__(**kwargs)
            self.cache = CacheIndex(evssm_cache, evssm_data_root)

        def __call__(self, results):
            original = list(results['img_filename'])
            resolved = [self.cache.resolve(path) for path in original]
            temporary = dict(results)
            temporary['img_filename'] = [path for path, _ in resolved]
            restored = super().__call__(temporary)
            for image, (_, record) in zip(restored['img'], resolved):
                if list(image.shape[:2]) != record['shape_hw']:
                    raise ValueError('EVSSM cache image dimensions do not match the original')
            # Keep original identifiers, camera order and calibrations in metadata.
            restored['img_filename'] = original
            restored['filename'] = original
            return restored


def configure_evssm(args, cfg, training=False):
    checkpoint = getattr(args, 'resume_from_checkpoint', None) if training else args.checkpoint_dir
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.is_file():
            checkpoint_path = checkpoint_path.parent
        if checkpoint_path.name == 'unet':
            checkpoint_path = checkpoint_path.parent
        checkpoint = str(checkpoint_path)
    saved = None
    if checkpoint and (Path(checkpoint) / 'evssm_config.json').is_file():
        saved = json.loads((Path(checkpoint) / 'evssm_config.json').read_text())
    if saved and not args.use_evssm:
        raise ValueError('This checkpoint requires --use_evssm and the same EVSSM cache')
    if not args.use_evssm:
        if args.evssm_cache or args.evssm_data_root:
            raise ValueError('--evssm_cache and --evssm_data_root require --use_evssm')
        return None
    standard_root = Path(__file__).resolve().parents[2] / 'data'
    args.evssm_cache = args.evssm_cache or str(standard_root / 'evssm_cache')
    args.evssm_data_root = args.evssm_data_root or str(standard_root / 'nuscenes')
    if training and checkpoint and not saved:
        raise ValueError('Cannot exactly resume a baseline/RAF run with EVSSM. Start a new run or use --pretrained_unet_checkpoint.')
    cache = CacheIndex(args.evssm_cache, args.evssm_data_root)
    if saved and saved['identity'] != cache.identity:
        raise ValueError('EVSSM checkpoint/cache identity mismatch')
    register_loader()
    replaced = [0]

    def walk(node):
        if isinstance(node, dict):
            if node.get('type') == 'LoadMultiViewImageFromFiles':
                node.update(type='LoadEVSSMMultiViewImageFromFiles',
                            evssm_cache=str(cache.root), evssm_data_root=str(cache.data_root))
                replaced[0] += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
    walk(cfg._cfg_dict if hasattr(cfg, '_cfg_dict') else cfg)
    if not replaced[0]:
        raise ValueError('No supported multi-camera image loader in the config')
    if training:
        args.output_dir = args.output_dir.rstrip('/') + '-evssm'
        args.tracker_run_name += '-evssm'
    print('EVSSM cache enabled: {} images, {}'.format(cache.image_count, cache.root), flush=True)
    return dict(version=1, method='EVSSM', identity=cache.identity,
                cache_root=str(cache.root), data_root=str(cache.data_root))


def save_evssm_config(config, directory):
    if config is not None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'evssm_config.json').write_text(json.dumps(config, indent=2) + '\n')
