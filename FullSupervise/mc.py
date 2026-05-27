import json
import multiprocessing
from collections import defaultdict
from typing import Dict

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def mc(box_per_image: np.ndarray):
    if len(box_per_image) < 2:
        return None
    num_box = len(box_per_image)
    mc_value = (np.abs(np.corrcoef(box_per_image)).sum() - num_box) / (num_box * (num_box - 1))
    return mc_value.item()


def coco_mc(coco_ann: str, process: int = 6):
    with open(coco_ann, "r") as f:
        ann_dict = json.load(f)

    bbox_per_image = defaultdict(list)
    for annotation in ann_dict["annotations"]:
        bbox_per_image[annotation["image_id"]].append(annotation["bbox"])

    dataset_mc = tqdm(
        multiprocessing.Pool(processes=process).imap(mc, bbox_per_image.values()),
        total=len(bbox_per_image),
        desc=f"Processing {coco_ann}"
    )
    return [t for t in dataset_mc if t is not None]


def plot_multiple_coco_mc(ann_paths: Dict[str, str], process: int = 6):
    plt.figure(figsize=(10, 6))
    for label, path in ann_paths.items():
        dataset_mc = coco_mc(path, process)
        sns.histplot(dataset_mc, stat="density", bins=80, kde=True, alpha=0.3, edgecolor=None, label=label)

    plt.legend()
    plt.xlabel("Macroscopic correlation")
    plt.ylabel("Density of images")
    plt.title("MC distribution across datasets")
    plt.tight_layout()
    plt.show()
    plt.savefig('mc.jpg', dpi=300)


# 🔧 替换为你的四个 COCO 注释文件路径
ann_files = {
    "3VT-A": "/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/3VT/c1/train/annotation.json",
    "4C-A": "/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/4C/c1/train/annotation.json",
    "3VT-B": "/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/3VT/c3/train/annotation.json",
    "4C-B": "/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/4C/c3/train/annotation.json",
    "EPV": "/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/EP/c1/output.json",
    "MMWHS-CT": "/media/Storage3/wlw/Relation-DETR/datasets/MMWHS/ct/train/annotations.json",
    "MMWHS-MRI": "/media/Storage3/wlw/Relation-DETR/datasets/MMWHS/mr/train/annotations.json",
}

# 🚀 绘图
plot_multiple_coco_mc(ann_files)
