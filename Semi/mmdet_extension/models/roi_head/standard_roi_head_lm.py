# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection

"""
standard roi_head for LabelMatch
"""
import torch

from mmdet.models.builder import HEADS
from mmdet.core import bbox2roi

from mmdet_extension.models.roi_head import StandardRoIHeadBase


@HEADS.register_module()
class StandardRoIHeadLM(StandardRoIHeadBase):
    def forward_train_step1(self,
                            x,
                            img_metas,
                            proposal_list,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_ignore=None,
                            gt_labels_ignore=None,
                            ):
        num_imgs = len(img_metas)
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], None, gt_labels[i])
            assign_result_ig = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes_ignore[i], None, gt_labels_ignore[i])
            sampling_result = self.bbox_sampler.sample_pos_ig(
                assign_result, assign_result_ig, proposal_list[i],
                gt_bboxes[i], gt_labels[i], gt_bboxes_ignore[i], gt_labels_ignore[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        return sampling_results

    def forward_train_step2(self,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels
                            ):
        losses = dict()
        rois = bbox2roi([res.bboxes for res in sampling_results])
        flag = torch.cat([res.ignore_flag for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets_lm(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        losses.update(bbox_results['loss_bbox'])
        scores = bbox_results['cls_score'][flag]
        label_index = torch.argmax(bbox_results['cls_score'], dim=1)
        counts_1_to_9 = torch.bincount(label_index[(label_index > 0) & (label_index < 10)], minlength=10)[1:]
        total_count = counts_1_to_9.sum().item()
        proportions = counts_1_to_9.float() / total_count
        # bboxes = torch.cat([res.bboxes for res in sampling_results])
        bboxes = bbox_results['bbox_pred']

        def bbox_area(bbox):
            return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

        needed_class = 1
        needed_bbox_indices = torch.nonzero(label_index == needed_class).squeeze(1)
        needed_bboxes = bboxes[needed_bbox_indices]
        needed_bbox_areas = bbox_area(needed_bboxes)
        average_area = needed_bbox_areas.mean()
        category_ratios = {}
        category_ratios[needed_class] = 1.0
        for class_id in range(10):  
            if class_id != needed_class:
                class_areas = bbox_area(bboxes[label_index == class_id])
                class_average_area = class_areas.mean() if class_areas.numel() > 0 else 0
                ratio = class_average_area / average_area if average_area > 0 else 0
                category_ratios[class_id] = ratio

        nonzero_indices = label_index.nonzero().squeeze()  # 获取标签不为 0 的索引
        nonzero_bboxes = bboxes[nonzero_indices]  # 根据索引获取边界框
        nonzero_labels = label_index[nonzero_indices]
        return losses, scores, proportions, category_ratios, nonzero_bboxes, nonzero_labels
