import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import argparse
import os

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)
    return images, annotations, categories

def compute_area_ratios(images, annotations, categories):
    category_ids = sorted(categories.keys())
    area_data = defaultdict(lambda: defaultdict(float))  # image_id -> category_id -> total area

    # 统计每张图中每个类别的总面积
    for img_id in tqdm(images):
        for ann in annotations[img_id]:
            cat_id = ann['category_id']
            _, _, w, h = ann['bbox']
            area = w * h
            area_data[img_id][cat_id] += area

    # 计算 log(area_a / area_b)
    ratio_dict = defaultdict(list)
    for img_id, cat_area in area_data.items():
        for i in category_ids:
            for j in category_ids:
                if i != j and i in cat_area and j in cat_area and cat_area[j] > 0:
                    ratio = np.log(cat_area[i] / cat_area[j])
                    key = (categories[i], categories[j])
                    ratio_dict[key].append(ratio)
    
    return ratio_dict

def plot_ratio_distribution(ratio_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for (cat_a, cat_b), ratios in ratio_dict.items():
        if len(ratios) < 10:
            continue
        plt.figure(figsize=(6, 4))
        sns.kdeplot(ratios, fill=True)
        plt.title(f"Log Area Ratio: {cat_a} / {cat_b}")
        plt.xlabel("log(area_a / area_b)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{cat_a}_vs_{cat_b}.png"))
        plt.close()

def plot_heatmap(ratio_dict, categories, save_path):
    cat_names = list(sorted(set([name for pair in ratio_dict for name in pair])))
    name_to_idx = {name: i for i, name in enumerate(cat_names)}
    mat = np.zeros((len(cat_names), len(cat_names)))
    for (a, b), values in ratio_dict.items():
        if len(values) > 0:
            mean_ratio = np.mean(values)
            mat[name_to_idx[a], name_to_idx[b]] = mean_ratio
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=cat_names, yticklabels=cat_names, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mean log(area_a / area_b)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_variance_heatmap(ratio_dict, categories, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    cat_names = list(sorted(set([name for pair in ratio_dict for name in pair])))
    name_to_idx = {name: i for i, name in enumerate(cat_names)}
    var_mat = np.zeros((len(cat_names), len(cat_names)))

    for (a, b), values in ratio_dict.items():
        if len(values) > 0:
            variance = np.var(values)
            var_mat[name_to_idx[a], name_to_idx[b]] = variance

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.4)  # 放大整体字体
    ax = sns.heatmap(
        var_mat,
        xticklabels=cat_names,
        yticklabels=cat_names,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"shrink": 0.8, "label": "Variance of log(area_a / area_b)"}
    )
    ax.set_title("Scale Variance Across Category Pairs", fontsize=20, pad=20)
    ax.set_xlabel("Category B", fontsize=16, labelpad=10)
    ax.set_ylabel("Category A", fontsize=16, labelpad=10)

    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', type=str, default='/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/3VT/c1/train/annotation.json', required=True, help="Path to COCO annotation json")
    parser.add_argument('--out_dir', type=str, default='ratio_output', help="Directory to save output plots")
    args = parser.parse_args()

    images, annotations, categories = load_coco_annotations(args.ann)
    ratio_dict = compute_area_ratios(images, annotations, categories)

    plot_variance_heatmap(ratio_dict, categories, os.path.join(args.out_dir, "heatmap_var_ratio.png"))

    print(f"Plotting {len(ratio_dict)} ratio distributions...")
    plot_ratio_distribution(ratio_dict, args.out_dir)
    plot_heatmap(ratio_dict, categories, os.path.join(args.out_dir, "heatmap_mean_ratio.png"))
