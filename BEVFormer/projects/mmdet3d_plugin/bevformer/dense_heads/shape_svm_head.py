# SPDX-License-Identifier: AGPL-3.0
"""Optional BEVFormer detection head with joint fine/coarse kernel SVMs."""
import torch
from mmdet.models import HEADS
from mmcv.runner import force_fp32
from .bevformer_head import BEVFormerHead
from projects.bevdiffuser.shape_modules import ShapeSVM


@HEADS.register_module()
class ShapeSVMHead(BEVFormerHead):
    def __init__(self, svm_options=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_classes != 10 or not self.loss_cls.use_sigmoid or self.as_two_stage:
            raise ValueError('ShapeSVMHead supports the ten-class, one-stage nuScenes head')
        self.shape_svm = ShapeSVM(self.embed_dims, **(svm_options or {}))

    def forward(self, *args, **kwargs):
        if kwargs.get('only_bev', False):
            return super().forward(*args, **kwargs)
        captured = []
        def capture(module, inputs, output):
            captured.append(output[1].permute(0, 2, 1, 3))
        hook = self.transformer.register_forward_hook(capture)
        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            hook.remove()
        fine, group = self.shape_svm(captured[0], outputs['all_bbox_preds'])
        outputs['all_cls_scores'] = self.shape_svm.fuse(outputs['all_cls_scores'], fine, group)
        outputs['svm_fine_scores'] = fine
        outputs['svm_group_scores'] = group
        return outputs

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts,
             gt_bboxes_ignore=None, img_metas=None):
        losses = super().loss(gt_bboxes_list, gt_labels_list, preds_dicts,
                              gt_bboxes_ignore=gt_bboxes_ignore, img_metas=img_metas)
        cls = preds_dicts['all_cls_scores'][-1]
        boxes = preds_dicts['all_bbox_preds'][-1]
        device = cls.device
        gt_boxes = [torch.cat((b.gravity_center, b.tensor[:, 3:]), dim=1).to(device)
                    for b in gt_bboxes_list]
        targets = self.get_targets(list(cls), list(boxes), gt_boxes, gt_labels_list,
                                   gt_bboxes_ignore)
        labels, weights = torch.cat(targets[0]), torch.cat(targets[1])
        losses['loss_shape_svm'] = self.shape_svm.objective(
            preds_dicts['svm_fine_scores'][-1], preds_dicts['svm_group_scores'][-1],
            labels, weights, cls.sigmoid().max(dim=-1).values)
        return losses
