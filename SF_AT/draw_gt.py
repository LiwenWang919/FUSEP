import cv2
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import os
from detectron2.config import get_cfg
from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer
from detectron2.engine import default_argument_parser, default_setup, launch
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.checkpoint import DetectionCheckpointer
from adapteacher.data.build import build_detection_test_loader


# ==== Predefined splits for Fetus-Dataset (COCO format) ===========

import colorsys

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg

def high_contrast_colors(n, s=0.9, v=0.9):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)
        # 将RGB值从0-1缩放到0-255，并四舍五入为整数
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        colors.append(rgb)
    return colors

# colors = high_contrast_colors(10)

colors = [
    (1.0, 0.0, 0.0, 1.0),  # 红色
    (0.0, 1.0, 0.0, 1.0),  # 绿色
    (0.0, 0.0, 1.0, 1.0),  # 蓝色
    (1.0, 1.0, 0.0, 1.0),  # 黄色
    (1.0, 0.0, 1.0, 1.0),  # 洋红色
    (0.0, 1.0, 1.0, 1.0),  # 青色
    (1.0, 0.5, 0.0, 1.0),  # 橙色
    (0.5, 0.0, 1.0, 1.0),  # 紫色
    (0.0, 0.5, 0.5, 1.0)   # 暗青色
]
colors_a = [
    (1.0, 0.0, 0.0, 0.3),  # 红色
    (0.0, 1.0, 0.0, 0.3),  # 绿色
    (0.0, 0.0, 1.0, 0.3),  # 蓝色
    (1.0, 1.0, 0.0, 0.3),  # 黄色
    (1.0, 0.0, 1.0, 0.3),  # 洋红色
    (0.0, 1.0, 1.0, 0.3),  # 青色
    (1.0, 0.5, 0.0, 0.3),  # 橙色
    (0.5, 0.0, 1.0, 0.3),  # 紫色
    (0.0, 0.5, 0.5, 0.3)   # 暗青色
]


def register_all_fetus():
    basepath = '/media/Storage1/Lvxg/One_Stage_Fetus_Object_Detection_Code_v3/Dataset_Fetus_Object_Detection/'
    SPLITS = {
    "fetus_4c_hos3_train": ('SF_AT/adapteacher/data/fetus_annotations_coco/4c/c3/train.json', basepath+'Hospital_3/four_chamber_heart'),
    "fetus_4c_hos2_train": ('SF_AT/adapteacher/data/fetus_annotations_coco/4c/c2/train/annotation.json', basepath+'Hospital_2/four_chamber_heart'),
    "fetus_4c_hos1_train": ('SF_AT/adapteacher/data/fetus_annotations_coco/4c/c1/train/annotation.json', basepath+'Hospital_1/four_chamber_heart'),
    "fetus_4c_hos1_val": ("CMT_AT/adapteacher/data/fetus_annotations_coco/4c/c1/val/annotation.json", basepath+'Hospital_1/four_chamber_heart'),
    "fetus_4c_hos1_test": ("SF_AT/adapteacher/data/fetus_annotations_coco/4c/c1/test/annotation.json", basepath+'Hospital_1/four_chamber_heart'),
    "fetus_4c_hos2_test": ("SF_AT/adapteacher/data/fetus_annotations_coco/4c/c2/test/annotation.json", basepath+'Hospital_2/four_chamber_heart'),
    "EP_public_train": ("/media/Storage1/Lvxg/EP_dataset/EP_public_annotations/train/annotation.json", "/media/Storage1/Lvxg/EP_dataset/EP_public_img"),
    "EP_public_test": ("/media/Storage1/Lvxg/EP_dataset/annotation.json", "/media/Storage1/Lvxg/EP_dataset/EP_public_img"),
    
    "EP_ours_train": ("/media/Storage1/Lvxg/EP_dataset/EP_fetus_annotations/train/annotation.json", "/media/Storage1/Lvxg/EP_dataset/EP_fetus"),
    "EP_ours_test": ("/media/Storage1/Lvxg/EP_dataset/annotation.json", "/media/Storage1/Lvxg/EP_dataset/EP_fetus"),
    }
    for key, (json_dir, img_dir) in SPLITS.items():
        register_coco_instances(key, {}, json_dir, img_dir)

register_all_fetus()
# dataset_dicts = DatasetCatalog.get("fetus_4c_hos3_train")
# metadata = MetadataCatalog.get("fetus_4c_hos3_train")
# metadata.thing_colors = COLOR
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     name = d["file_name"].split('/')[-1]
#     cv2.imwrite(f'./visual/gt/{name}', img)
#     visualizer = Visualizer(img[:,:,::-1], metadata=metadata, scale=1)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imwrite(f'./visual/gt_with_box/{name}', out.get_image())

#######

DRAW = 'GT' # 'GT' or 'Pred'

if DRAW =='GT':
    fname = 'EP_ours_test'
    dataset_dicts = DatasetCatalog.get(fname)
    metadata = MetadataCatalog.get(fname)
    for d in random.sample(dataset_dicts, 5): # random.sample(dataset_dicts, 5)
        image = cv2.imread(d["file_name"])
        name = d["file_name"].split('/')[-1]
        v = Visualizer(image[:, :, ::-1], metadata)
        for anno in d["annotations"]:
            bbox = anno["bbox"]
            category_id = anno["category_id"]
            color = colors[category_id]
            color_a = colors_a[category_id]

            # 转换边界框格式从 [x,y,width,height] 到 [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

            # 绘制边界框和类别标签
            v.draw_box([x1, y1, x2, y2], edge_color=color)
            # category_name = metadata.thing_classes[category_id]
            # v.draw_text(category_name, (x1, y1), color=color_a)

        result_image = v.get_output().get_image()[:, :, ::-1]
        # cv2.imwrite(f'vis_EP/public/gt_{name}', image)
        # cv2.imwrite(f'vis_EP/public/box_{name}', result_image)
        cv2.imwrite(f'box_{name}', result_image)

elif DRAW =='Pred':
    
    fname = 'fetus_4c_hos2_test'
    path = 'visual/ours1-2'
    cfg_path = "/media/Storage1/Lvxg/CMT/SF_AT/configs/sfda_draw.yaml" # sfda_test
    weights_path = "/media/Storage1/Lvxg/CMT/output/output2024/output_acmmm_final1_/model_0015999_79.25.pth"


    dataset_dicts = DatasetCatalog.get(fname)
    metadata = MetadataCatalog.get(fname)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    # cfg = setup(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path

    predictor = DefaultPredictor(cfg)

    # Trainer = ATeacherTrainer
    # model = Trainer.build_model(cfg)
    # model_teacher = Trainer.build_model(cfg)
    # ensem_ts_model = EnsembleTSModel(model_teacher, model)

    # DetectionCheckpointer(
    #     model_teacher, save_dir=cfg.OUTPUT_DIR
    # ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    # if cfg.TEST.EVAL_STU:
    #     res = Trainer.test(cfg, ensem_ts_model.modelStudent)
    # else:
    #     res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

    # data_loader = build_detection_test_loader(cfg, fname)
    # for idx, inputs in enumerate(data_loader):
    #     predictor = model_teacher(inputs)

    for d in dataset_dicts:
        # 读取图像
        img = cv2.imread(d["file_name"])
        name = d["file_name"].split('/')[-1]
        # 进行预测
        # predictor = model_teacher(img)
        outputs = predictor(img)
        
        # 创建一个Visualizer对象，并使用预测结果可视化图像
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None

        for i, box in enumerate(boxes):
            # color = custom_colors[i % len(custom_colors)]  # 循环使用自定义颜色列表
            v.draw_box(box, edge_color=colors)

        out = v.get_image()[:, :, ::-1]
        
        # 可视化结果
        cv2.imwrite(os.path.join(path, f'box_{name}'), out)