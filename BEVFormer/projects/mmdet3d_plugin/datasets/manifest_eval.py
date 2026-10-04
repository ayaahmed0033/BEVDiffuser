"""nuScenes detection metrics on an explicit custom validation manifest.

Uses the devkit's standard matching, AP/TP metrics and box filtering, but never
loads an official scene split or drops missing prediction samples silently.
"""
import json
import os
import time
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetrics, DetectionMetricDataList
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.utils import category_to_detection_name
from projects.custom_split import read_manifest, manifest_hash, require_tokens


class ManifestDetectionEval:
    def __init__(self, nusc, config, result_path, output_dir, manifest_path,
                 data_infos, expected_hash, verbose=True, overlap_test=False):
        if overlap_test:
            raise ValueError('Custom manifest evaluation supports standard detection only; overlap_test must be False')
        self.nusc, self.cfg = nusc, config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        manifest = read_manifest(manifest_path, nusc)
        self.digest = manifest_hash(manifest)
        if self.digest != expected_hash:
            raise ValueError('Manifest changed after info conversion. Regenerate info files.')
        self.tokens = manifest['val']['sample_tokens']
        require_tokens([r['token'] for r in data_infos], self.tokens, 'Evaluation dataset')
        self.pred_boxes, self.meta = load_prediction(result_path, config.max_boxes_per_sample,
                                                     DetectionBox, verbose=verbose)
        require_tokens(self.pred_boxes.sample_tokens, self.tokens, 'Predictions')
        gt = EvalBoxes()
        for token in self.tokens:
            boxes = []
            sample = nusc.get('sample', token)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                name = category_to_detection_name(ann['category_name'])
                if name is None:
                    continue
                attrs = ann['attribute_tokens']
                if len(attrs) > 1:
                    raise ValueError('Multiple attributes on annotation ' + ann_token)
                attr = nusc.get('attribute', attrs[0])['name'] if attrs else ''
                boxes.append(DetectionBox(sample_token=token, translation=tuple(ann['translation']),
                    size=tuple(ann['size']), rotation=tuple(ann['rotation']),
                    velocity=tuple(nusc.box_velocity(ann_token)[:2]),
                    num_pts=ann['num_lidar_pts'] + ann['num_radar_pts'],
                    detection_name=name, detection_score=-1.0, attribute_name=attr))
            gt.add_boxes(token, boxes)  # retain empty frames as required evaluation samples
        def filtered(boxes):
            boxes = add_center_dist(nusc, boxes)
            # Older devkits cannot infer the type if every sample has zero boxes.
            if any(boxes[token] for token in boxes.sample_tokens):
                boxes = filter_eval_boxes(nusc, boxes, config.class_range, verbose=verbose)
            return boxes
        self.pred_boxes = filtered(self.pred_boxes)
        self.gt_boxes = filtered(gt)

    def main(self, plot_examples=0, render_curves=False):
        if plot_examples or render_curves:
            raise ValueError('This custom evaluator exports metrics only; plot_examples=0, render_curves=False required')
        start = time.time()
        details = DetectionMetricDataList()
        metrics = DetectionMetrics(self.cfg)
        for name in self.cfg.class_names:
            for distance in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, name, self.cfg.dist_fcn_callable, distance)
                details.set(name, distance, md)
                metrics.add_label_ap(name, distance, calc_ap(md, self.cfg.min_recall, self.cfg.min_precision))
            for metric in TP_METRICS:
                if (name == 'traffic_cone' and metric in ('attr_err','vel_err','orient_err')) or (name == 'barrier' and metric in ('attr_err','vel_err')):
                    value = np.nan
                else:
                    value = calc_tp(details[(name, self.cfg.dist_th_tp)], self.cfg.min_recall, metric)
                metrics.add_label_tp(name, metric, value)
        metrics.add_runtime(time.time() - start)
        summary = metrics.serialize()
        summary['meta'] = self.meta
        summary['custom_protocol'] = {'split': 'val', 'sample_count': len(self.tokens),
                                      'manifest_sha256': self.digest, 'official_benchmark': False}
        for filename, data in [('metrics_summary.json', summary), ('metrics_details.json', details.serialize())]:
            with open(os.path.join(self.output_dir, filename), 'w') as f:
                json.dump(data, f, indent=2)
        print('Custom validation: %d samples, mAP %.4f, NDS %.4f' %
              (len(self.tokens), summary['mean_ap'], summary['nd_score']))
        return summary
