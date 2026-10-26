# SPDX-License-Identifier: AGPL-3.0
"""On-the-fly, pre-normalization camera grayscale augmentation."""
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomGrayscaleMultiView:
    def __init__(self, probability=0.3):
        if not 0 <= probability <= 1:
            raise ValueError('Grayscale probability must be between zero and one')
        self.probability = probability

    def __call__(self, results):
        # One decision across six cameras in this frame preserves view consistency.
        # No images are saved, renamed, resized, or assigned a different camera.
        if np.random.random() >= self.probability:
            return results
        images = []
        for image in results['img']:
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError('Expected three-channel BGR camera images before normalization')
            # LoadMultiViewImageFromFiles supplies BGR. Retain three channels.
            gray = (0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2])
            images.append(np.repeat(gray[..., None], 3, axis=-1).astype(image.dtype))
        results['img'] = images
        return results


def enable_grayscale(train_cfg, probability):
    """Insert immediately before normalization, after existing photometric jitter."""
    if 'pipeline' not in train_cfg:
        if 'dataset' in train_cfg:
            enable_grayscale(train_cfg['dataset'], probability)
            return
        raise ValueError('Training dataset has no pipeline')
    pipeline = train_cfg['pipeline']
    if any(p.get('type') == 'RandomGrayscaleMultiView' for p in pipeline):
        raise ValueError('Grayscale augmentation is already configured')
    for i, item in enumerate(pipeline):
        if item.get('type') == 'NormalizeMultiviewImage':
            pipeline.insert(i, dict(type='RandomGrayscaleMultiView', probability=probability))
            return
    raise ValueError('Expected NormalizeMultiviewImage in the training pipeline')
