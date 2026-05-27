# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
hooks for LabelMatch
"""
import shutil
import os.path as osp
import numpy as np

import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core.evaluation import EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


@HOOKS.register_module()
class LabelMatchHook(Hook):
    def __init__(self, cfg):
        rank, world_size = get_dist_info()
        distributed = world_size > 1
        samples_per_gpu = cfg.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.pipeline = replace_ImageToTensor(cfg.data.pipeline)
        # random select 10000 image as reference image (in order to save the inference time)
        dataset = build_dataset(cfg.data, dict(test_mode=True))
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        self.CLASSES = dataset.CLASSES
        boxes_per_image_gt, cls_ratio_gt, area_info = self.get_data_info(cfg.label_file)
        self.area_info = area_info
        file = cfg.data['ann_file']
        eval_cfg = cfg.get('evaluation', {})
        manual_prior = cfg.get('manual_prior', None)
        if manual_prior:  # manual setting the boxes_per_image and cls_ratio
            boxes_per_image_gt = manual_prior.get('boxes_per_image', boxes_per_image_gt)
            cls_ratio_gt = manual_prior.get('cls_ratio', cls_ratio_gt)
        min_thr = cfg.get('min_thr', 0.05)  # min cls score threshold for ignore
        potential_positive = len(dataset) * boxes_per_image_gt * cls_ratio_gt
        if distributed:
            self.eval_hook = LabelMatchDistEvalHook(
                file, dataloader, potential_positive, boxes_per_image_gt, cls_ratio_gt, area_info, min_thr, **eval_cfg)
        else:
            self.eval_hook = LabelMatchEvalHook(
                file, dataloader, potential_positive, boxes_per_image_gt, cls_ratio_gt, area_info, min_thr, **eval_cfg)

    def get_data_info(self, json_file):
        """get information from labeled data"""
        info = mmcv.load(json_file)
        area_info = self.analyze_area(info)
        id2cls = {}
        total_image = len(info['images'])
        for value in info['categories']:
            print(id2cls)
            print(value)
            id2cls[value['id']] = self.CLASSES.index(value['name'])
        cls_num = [0] * len(self.CLASSES)
        for value in info['annotations']:
            cls_num[id2cls[value['category_id']]] += 1
        cls_num = [max(c, 1) for c in cls_num]  # for some cls not select, we set it 1 rather than 0
        total_boxes = sum(cls_num)
        cls_ratio_gt = np.array([c / total_boxes for c in cls_num])
        boxes_per_image_gt = total_boxes / total_image
        logger = get_root_logger()
        info = ' '.join([f'({v:.4f}-{self.CLASSES[i]})' for i, v in enumerate(cls_ratio_gt)])
        logger.info(f'boxes per image (label data): {boxes_per_image_gt}')
        logger.info(f'class ratio (label data): {info}')
        return boxes_per_image_gt, cls_ratio_gt, area_info

    def analyze_area(self, info):
        # 计算每个bounding box的面积
        def bbox_area(bbox):
            xmin, ymin, xmax, ymax = bbox
            return (xmax - xmin) * (ymax - ymin)

        # 按分类分组并计算总面积和总数量
        category_areas = {}
        category_counts = {}
        fixed_category_area = 0.0

        for item in info['annotations']:
            category_id = item['category_id']
            bbox = item['bbox']
            area = bbox_area(bbox)
            
            if category_id == 1:
                fixed_category_area = fixed_category_area + area
                category_counts[category_id] = category_counts.get(category_id, 0) + 1
            else:
                category_areas[category_id] = category_areas.get(category_id, 0) + area
                category_counts[category_id] = category_counts.get(category_id, 0) + 1

        # 计算每个分类的平均面积
        category_avg_areas = {}
        fixed_category_area = fixed_category_area / category_counts.get(1, 0)
        for category_id, total_area in category_areas.items():
            count = category_counts.get(category_id, 0)  # 获取分类数量，如果不存在则默认为0
            avg_area = total_area / count if count > 0 else 0  # 计算平均面积，避免除以0错误
            category_avg_areas[category_id] = avg_area

        # 将category_id为1的分类的比例值固定为1，其他分类的比例值为该分类的平均面积除以category_id为1的分类的平均面积
        category_ratios = {}
        for category_id, avg_area in category_avg_areas.items():
            if category_id == 1:
                category_ratios[category_id] = 1.0
            else:
                ratio = avg_area / fixed_category_area if fixed_category_area is not None else 0  # 计算比值，避免除以0错误
                category_ratios[category_id] = ratio
        category_ratios[1] = 1.0

        return category_ratios

    def before_train_epoch(self, runner):
        self.eval_hook.before_train_epoch(runner)

    def after_train_epoch(self, runner):
        self.eval_hook.after_train_epoch(runner)

    def after_train_iter(self, runner):
        self.eval_hook.after_train_iter(runner)

    def before_train_iter(self, runner):
        self.eval_hook.before_train_epoch(runner)


class LabelMatchEvalHook(EvalHook):
    def __init__(self,
                 file,
                 dataloader,
                 potential_positive,
                 boxes_per_image_gt,
                 cls_ratio_gt,
                 area_info,
                 min_thr,
                 **eval_kwargs
                 ):
        super().__init__(dataloader, **eval_kwargs)
        self.file = file
        self.dst_root = None
        self.initial_epoch_flag = True

        self.potential_positive = potential_positive
        self.boxes_per_image_gt = boxes_per_image_gt
        self.cls_ratio_gt = cls_ratio_gt
        self.min_thr = min_thr
        self.dataloader = dataloader
        self.area_info = area_info

        self.CLASSES = self.dataloader.dataset.CLASSES

    def before_train_epoch(self, runner):
        if not self.initial_epoch_flag:
            return
        if self.dst_root is None:
            self.dst_root = runner.work_dir
        interval_temp = self.interval
        self.interval = 1
        if self.by_epoch:
            self.after_train_epoch(runner)
        else:
            self.after_train_iter(runner)
        self.initial_epoch_flag = False
        self.interval = interval_temp
        runner.model.module.boxes_per_image_gt = self.boxes_per_image_gt
        runner.model.module.cls_ratio_gt = self.cls_ratio_gt
        runner.model.module.area_info = self.area_info

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.evaluation_flag(runner):
            return
        self.update_cls_thr(runner)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        self.update_cls_thr(runner)

    def do_evaluation(self, runner):
        from mmdet_extension.apis.test import single_gpu_test
        self.dataloader.dataset.shuffle_data_info()  # random shuffle
        results = single_gpu_test(runner.model.module.ema_model, self.dataloader)
        return results

    def update_cls_thr(self, runner):
        percent = runner.model.module.percent  # percent as positive
        results = self.do_evaluation(runner)
        cls_thr, cls_thr_ig = self.eval_score_thr(results, percent)
        runner.model.module.cls_thr = cls_thr
        runner.model.module.cls_thr_ig = cls_thr_ig

    def eval_score_thr(self, results, percent):
        # ACT dynamic thr
        score_list = [[] for _ in self.CLASSES]
        for result in results:
            for cls, r in enumerate(result):
                score_list[cls].append(r[:, -1])
        score_list = [np.concatenate(c) for c in score_list]
        score_list = [np.zeros(1) if len(c) == 0 else np.sort(c)[::-1] for c in score_list]
        cls_thr = [0] * len(self.CLASSES)
        cls_thr_ig = [0] * len(self.CLASSES)
        for i, score in enumerate(score_list):
            cls_thr[i] = max(0.05, score[min(int(self.potential_positive[i] * percent), len(score) - 1)])
            # NOTE: original use 0.05, for UDA, we change to 0.001
            cls_thr_ig[i] = max(self.min_thr, score[min(int(self.potential_positive[i]), len(score) - 1)])
        logger = get_root_logger()
        logger.info(f'current percent: {percent}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr)])
        logger.info(f'update score thr (positive): {info}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr_ig)])
        logger.info(f'update score thr (ignore): {info}')
        return cls_thr, cls_thr_ig


class LabelMatchDistEvalHook(LabelMatchEvalHook):
    def __init__(self,
                 file,
                 dataloader,
                 potential_positive,
                 boxes_per_image_gt,
                 cls_ratio_gt,
                 area_info,
                 min_thr,
                 tmpdir=None,
                 gpu_collect=False,
                 broadcast_bn_buffer=True,
                 **eval_kwargs
                 ):
        super().__init__(file, dataloader, potential_positive, boxes_per_image_gt, cls_ratio_gt, min_thr, **eval_kwargs)
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect
        self.area_info = area_info

    def _broadcast_bn_buffer(self, model):
        if self.broadcast_bn_buffer:
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

    def do_evaluation(self, runner):
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(runner.model.module.ema_model)
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        from mmdet_extension.apis.test import multi_gpu_test
        self.dataloader.dataset.shuffle_data_info()
        results = multi_gpu_test(runner.model.module.ema_model, self.dataloader,
                                 tmpdir=tmpdir, gpu_collect=self.gpu_collect)
        return results

    def update_cls_thr(self, runner):
        percent = runner.model.module.percent  # percent as positive
        results = self.do_evaluation(runner)
        tmpdir = './tmp_file'
        tmpfile = osp.join(tmpdir, 'tmp.pkl')
        if runner.rank == 0:
            cls_thr, cls_thr_ig = self.eval_score_thr(results, percent)
            mmcv.mkdir_or_exist(tmpdir)
            mmcv.dump((cls_thr, cls_thr_ig), tmpfile)
        dist.barrier()
        cls_thr, cls_thr_ig = mmcv.load(tmpfile)
        dist.barrier()
        if runner.rank == 0:
            shutil.rmtree(tmpdir)
        runner.model.module.cls_thr = cls_thr
        runner.model.module.cls_thr_ig = cls_thr_ig
