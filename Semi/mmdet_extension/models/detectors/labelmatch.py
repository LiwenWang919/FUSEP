# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
LabelMatch
"""
import numpy as np

import torch

from mmcv.runner.dist_utils import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core.bbox.iou_calculators import iou2d_calculator
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.models.builder import DETECTORS

from mmdet_extension.models.detectors import SemiTwoStageDetector
from mmdet_extension.models.loss import SinkhornDistance
import torch.nn.functional as F
from mmdet_extension.models.utils.boxlist import BoxList
from mmdet_extension.models.utils.graph_config import _C as graph_opt
from mmdet_extension.models.loss.graph_matching_mbqu_frcnn import build_graph_matching_head
from mmdet_extension.models.utils.config import opt


@DETECTORS.register_module(name='LabelMatch', force=True)
class LabelMatch(SemiTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 classes=None,
                 # config
                 cfg=dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained,
                         ema_config=ema_config, ema_ckpt=ema_ckpt, classes=classes)
        self.debug = cfg.get('debug', False)
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0

        # hyper-parameter: fixed
        self.tpt = cfg.get('tpt', 0.5)
        self.tps = cfg.get('tps', 1.0)
        self.momentum = cfg.get('momentum', 0.996)
        self.weight_u = cfg.get('weight_u', 2.0)

        # adat
        score_thr = cfg.get('score_thr', 0.9)  # if not use ACT, will use this hard thr
        self.cls_thr = [0.9 if self.debug else score_thr] * self.num_classes
        self.cls_thr_ig = [0.2 if self.debug else score_thr] * self.num_classes
        self.percent = cfg.get('percent', 0.2)

        # mining
        self.use_mining = cfg.get('use_mining', True)
        self.reliable_thr = cfg.get('reliable_thr', 0.8)
        self.reliable_iou = cfg.get('reliable_iou', 0.8)

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_ig = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)
        self.pseudo_num_tp_ig = np.zeros(self.num_classes)
        self.pseudo_num_mining = np.zeros(self.num_classes)

        self.opt = opt

        self.sinkhornDistance = SinkhornDistance(0.001, 120)

        self.graph_matching = build_graph_matching_head(graph_opt, self.opt.out_channel)
        self.graph_matching.train()

        self.C = 20
        self.permution_queue = QueueWithBuffer2D(max_size=self.C, height=300, width=300)

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        # NOTE here for core code
        device = img.device
        self.graph_matching = self.graph_matching.to(device)
        _, _, h, w = img_unlabeled_1.shape
        self.image_num += len(img_metas_unlabeled)
        self.update_ema_model(self.momentum)
        self.cur_iter += 1
        self.analysis()  # record the information in the training
        # # ---------------------label data---------------------
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        features_labeled = self.extract_feat(img)
        losses = self.parse_loss(losses)
        # # -------------------unlabeled data-------------------
        bbox_transform, bbox_transform_1 = [], []
        for img_meta, img_meta_1 in zip(img_metas_unlabeled, img_metas_unlabeled_1):
            bbox_transform.append(img_meta.pop('bbox_transform'))
            bbox_transform_1.append(img_meta_1.pop('bbox_transform'))
        # create pseudo label
        ## teacher ema_model是指teacher
        bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True
        )
        teacher_proportion = self.calculate_proportion(bbox_results)
        mean_proportion = torch.tensor([sum(x) / len(x) for x in zip(*teacher_proportion)]).to(device)
        area_pro = self.ana_area(bbox_results)
        ## ACT 根据weak分支预测的bbox ig是指uncertain
        gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred = \
            self.create_pseudo_results(
                img_unlabeled_1, bbox_results, bbox_transform_1, device,
                gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
            )
        
        self.visual_offline(img_unlabeled, gt_bboxes_pred, gt_labels_pred, img_metas_unlabeled)
                            # boxes_ignore_list=gt_bboxes_ig_pred)
        # training on unlabeled data
        losses_unlabeled, proportions_unlabeled, area_ratios_unlabeled, features_unlabeled, gt_bboxes_unlabeled_graph, gt_labels_unlabeled_graph = self.training_unlabeled(
            img_unlabeled_1, img_metas_unlabeled_1, bbox_transform_1,
            img_unlabeled, img_metas_unlabeled, bbox_transform,
            gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred
        )
        standard_proportion = torch.Tensor([0.1 ,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]).to(device)
        standara_area_proportion = self.area_info
        loss_proportion = torch.sum(torch.abs(mean_proportion - proportions_unlabeled)).unsqueeze(0)
            # + torch.sum(torch.abs(mean_proportion - standard_proportion)).unsqueeze(0) \
            # + torch.sum(torch.abs(proportions_unlabeled - standard_proportion)).unsqueeze(0)
        # loss_proportion[0] = loss_proportion[0]/3
        del area_ratios_unlabeled[0]
        # 初始化总和
        total_sum = 0
        # 计算两两字典之间键的差值的绝对值总和
        def compute_absolute_difference_sum(dict_a, dict_b):
            abs_diff_sum = 0
            for key in dict_a:
                if key in dict_b:
                    abs_diff_sum += abs(dict_a[key] - dict_b[key])
            return abs_diff_sum

        # 计算两两字典之间的绝对值总和
        total_sum += compute_absolute_difference_sum(area_ratios_unlabeled, self.area_info)
        total_sum += compute_absolute_difference_sum(self.area_info, area_pro)
        total_sum += compute_absolute_difference_sum(area_pro, area_ratios_unlabeled)
        loss_area = torch.tensor(total_sum/100)
        losses_unlabeled = self.parse_loss(losses_unlabeled)

        ### Graph Matching
        targets_src = []
        for box, label in zip(gt_bboxes, gt_labels):
            targets = BoxList(box, (img.shape[2], img.shape[3]), mode='xyxy')
            targets.fields['labels'] = label
            targets_src.append(targets)


        targets_tgt = []
        gt_bboxes_unlabeled_graph = [gt_bboxes_unlabeled_graph]
        gt_labels_unlabeled_graph = [gt_labels_unlabeled_graph]
        for box, label in zip(gt_bboxes_unlabeled_graph, gt_labels_unlabeled_graph):
            targets = BoxList(box, (img_unlabeled.shape[2], img_unlabeled.shape[3]), mode='xyxy')
            targets.fields['labels'] = label
            targets_tgt.append(targets)

        features_labeled = list(features_labeled)
        features_unlabeled = list(features_unlabeled)


        (_, _), middle_head_loss, loss_cont, loss_sink, M_p = \
                    self.graph_matching(None, (features_labeled, features_unlabeled), targets=(targets_src, targets_tgt))
        self.permution_queue.enqueue(M_p)
        per_loss = self.permution_queue.compute_consistency_loss() / self.C
        # loss_graph = middle_head_loss + loss_cont + loss_sink
        # loss_graph = torch.tensor(middle_head_loss + loss_cont + loss_sink)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            else:
                losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        # extra info for analysis
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num.sum() / self.image_num]).to(device),
            'pseudo_num_ig': torch.Tensor([self.pseudo_num_ig.sum() / self.image_num]).to(device),
            'pseudo_num_mining': torch.Tensor([self.pseudo_num_mining.sum() / self.image_num]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / self.pseudo_num.sum()]).to(device),
            'pseudo_num ig(acc)': torch.Tensor([self.pseudo_num_tp_ig.sum() / (self.pseudo_num_ig.sum() + 1e-10)]).to(
                device),
        }
        losses.update({'loss_proportion': loss_proportion})
        losses.update({'loss_area': loss_area})
        losses.update({'loss_graph': torch.tensor(middle_head_loss['node_loss'] + middle_head_loss['mat_loss_aff'])})
        losses.update({'per_loss': torch.tensor(per_loss)})
        losses.update(extra_info)
        # self.visual_offline(img, gt_bboxes_pred, gt_labels_pred, img_metas,)
        return losses

    # # ---------------------------------------------------------------------------------
    # # training on unlabeled data
    # # ---------------------------------------------------------------------------------
    def training_unlabeled(self, img, img_metas, bbox_transform,
                           img_t, img_metas_t, bbox_transform_t,
                           gt_bboxes, gt_labels, gt_bboxes_ig, gt_labels_ig):
        losses = dict()
        x = self.extract_feat(img)
        # rpn loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        gt_bboxes_cmb = [torch.cat([a, b]) for a, b in zip(gt_bboxes, gt_bboxes_ig)]
        ## student pred
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x, img_metas, gt_bboxes_cmb, gt_labels=None, proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)
        # roi loss ## student pred sample
        sampling_results = self.roi_head.forward_train_step1(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ig, gt_labels_ig
        )
        ig_boxes = [torch.cat([ig, res.ig_bboxes])
                    for ig, res in zip(gt_bboxes_ig, sampling_results)]
        ig_len = [len(ig) for ig in gt_bboxes_ig]
        for i in range(len(img_metas)):
            ig_boxes[i] = self.rescale_bboxes(ig_boxes[i], img_metas[i], bbox_transform_t[i])
        ignore_boxes_t = [b[:l] for l, b in zip(ig_len, ig_boxes)]
        ig_boxes = [b[l:] for l, b in zip(ig_len, ig_boxes)]
        with torch.no_grad():
            ema_model = self.ema_model.module
            x_t = ema_model.extract_feat(img_t)
            det_bboxes_t, det_labels_t = ema_model.roi_head.simple_test_bboxes_base(
                x_t, img_metas_t, ig_boxes)
            cls_scores_t = [torch.softmax(l / self.tpt, dim=-1) for l in det_labels_t]
            det_labels_t = [torch.softmax(l, dim=-1) for l in det_labels_t]
        
        roi_losses, cls_scores, proportions, category_ratios, bboxes, labels = self.roi_head.forward_train_step2(
            x, sampling_results, gt_bboxes, gt_labels)
        losses.update(roi_losses)
        # proposal based learning
        weight = torch.cat([1 - res.ig_reg_weight for res in sampling_results])
        cls_scores_t = torch.cat(cls_scores_t)
        cls_scores = torch.softmax(cls_scores / self.tps, dim=-1)
        if len(cls_scores) > 0:
            avg_factor = len(img_metas) * self.roi_head.bbox_sampler.num
            losses_cls_ig = (-cls_scores_t * torch.log(cls_scores)).sum(-1)
            losses_cls_ig = (losses_cls_ig * weight).sum() / avg_factor
        else:
            losses_cls_ig = cls_scores.sum()  # 0
        losses.update({'losses_cls_ig': losses_cls_ig})
        return losses, proportions, category_ratios, x, bboxes, labels

    # # ---------------------------------------------------------------------------------
    # # create pseudo labels
    # # ---------------------------------------------------------------------------------
    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        gt_bboxes_ig_pred, gt_labels_ig_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        for b, result in enumerate(bbox_results):
            bboxes, labels = [], []
            bboxes_ig, labels_ig = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox / scale_factor
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag_pos = r[:, -1] >= self.cls_thr[cls]
                flag_ig = (r[:, -1] >= self.cls_thr_ig[cls]) & (~flag_pos)
                bboxes.append(r[flag_pos][:, :4])
                bboxes_ig.append(r[flag_ig][:, :4])
                labels.append(label[flag_pos])
                labels_ig.append(label[flag_ig])
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    overlap = bbox_overlaps(bboxes[-1], gt_bbox_scale[gt_label == cls])
                    iou = overlap.max(-1)
                    self.pseudo_num_tp[cls] += (iou > 0.5).sum()
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes_ig[-1]) > 0:
                    overlap = bbox_overlaps(bboxes_ig[-1], gt_bbox_scale[gt_label == cls])
                    iou = overlap.max(-1)
                    self.pseudo_num_tp_ig[cls] += (iou > 0.5).sum()
                self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
                self.pseudo_num_ig[cls] += len(bboxes_ig[-1])
            bboxes = np.concatenate(bboxes)
            bboxes_ig = np.concatenate(bboxes_ig)
            bboxes_concat = np.r_[bboxes, bboxes_ig]
            labels = np.concatenate(labels)
            labels_ig = np.concatenate(labels_ig)
            for bf in box_transform[b]:
                bboxes_concat, labels = bf(bboxes_concat, labels)
            bboxes, bboxes_ig = bboxes_concat[:len(bboxes)], bboxes_concat[len(bboxes):]
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
            gt_bboxes_ig_pred.append(torch.from_numpy(bboxes_ig).float().to(device))
            gt_labels_ig_pred.append(torch.from_numpy(labels_ig).long().to(device))
        return gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred

    # # -----------------------------analysis function------------------------------
    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_ig = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                                in zip(self.CLASSES, self.pseudo_num_ig, self.pseudo_num_tp_ig)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo ig: {info_ig}')
            logger.info(f'pseudo gt: {info_gt}')
            if self.use_mining:
                info_mining = ' '.join([f'{a}' for a in self.pseudo_num_mining])
                logger.info(f'pseudo mining: {info_mining}')

    def calculate_proportion(self, arrays):
        total_lengths = [sum(len(subarr) for subarr in arr) for arr in arrays]
        for i,length in enumerate(total_lengths):
            if length == 0:
                total_lengths[i] = 1
        proportions = [[len(subarr) / total_lengths[i] for subarr in arr] for i, arr in enumerate(arrays)]
        return proportions
    
    def ana_area(self, info):
        # 计算每个bounding box的面积
        def bbox_area(bbox):
            return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

        # 初始化分类面积字典和分类计数字典
        category_areas = {}
        category_counts = {}

        # 遍历数据列表
        for index, sublist in enumerate(info, start=0):
            for array_index, array in enumerate(sublist, start=0):
                # 获取分类ID
                category_id = array_index + 1
                
                # 如果数组为空，则跳过
                if array.size == 0:
                    total_area = 0
                
                # 计算面积
                areas = bbox_area(array)
                total_area = np.sum(areas)
                
                # 更新分类面积和计数
                category_areas[category_id] = category_areas.get(category_id, 0) + total_area
                category_counts[category_id] = category_counts.get(category_id, 0) + array.shape[0]

        # 计算每个分类的平均面积
        category_avg_areas = {}
        for category_id, total_area in category_areas.items():
            count = category_counts.get(category_id, 0)
            avg_area = total_area / count if count > 0 else 0
            category_avg_areas[category_id] = avg_area

        # 获取category_id为1的分类的平均面积
        fixed_category_area = category_avg_areas.get(1, 0)

        # 计算其他分类相对于category_id为1的分类的平均面积比值
        category_ratios = {}
        for category_id, avg_area in category_avg_areas.items():
            if category_id == 1:
                category_ratios[category_id] = 1.0
            else:
                ratio = avg_area / fixed_category_area if fixed_category_area > 0 else 0
                category_ratios[category_id] = ratio

        return category_ratios
    
    def constructGraph(self, proposal):
        self.adj_num = 21
        self.child_num = 4
        self.act_feat_dim = 1024
        activity_fts = proposal
        activity_fts = torch.unsqueeze(activity_fts, 0)
        batch_size = activity_fts.size()[0]

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous()

        # cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        act_cos_sim_mat = dot_product_mat / len_mat

        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num)
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0])
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)):
            mask_mat_var[row * self.adj_num : (row + 1) * self.adj_num, row * self.adj_num : (row + 1) * self.adj_num] \
                = mask

        act_adj_mat = mask_mat_var * act_cos_sim_mat

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat)
        return act_adj_mat


class QueueWithBuffer2D:
    def __init__(self, max_size, height, width):
        self.max_size = max_size
        self.height = height
        self.width = width
        self.size = 0
        self.buffer = torch.zeros((max_size, height, width), dtype=torch.float32)

    def enqueue(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Input value must be a PyTorch tensor")
        if value.shape != (self.height, self.width):
            raise ValueError("Input tensor shape must match the queue size")

        if self.size == self.max_size:
            co_buffer = self.buffer[1:].clone()
            self.buffer[:-1] = co_buffer
            del co_buffer
            self.buffer[-1] = value
        else:
            self.buffer[self.size] = value
            self.size += 1

    def compute_consistency_loss(self):
        loss = 0.0
        queue_c = self.buffer.clone()
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                matrix1 = queue_c[i]
                matrix2 = queue_c[j]
                loss += 1 - (matrix1 * matrix2).sum() / (self.height * self.width)
        del queue_c
        return loss
        