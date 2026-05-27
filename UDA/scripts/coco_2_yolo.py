import os
import json
import shutil
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return [
        x_center / img_width,
        y_center / img_height,
        width / img_width,
        height / img_height
    ]

def get_split_from_filename(filename):
    """
    从文件名推断当前是 train / val / test，例如：instances_train.json -> train
    """
    name = os.path.basename(filename).lower()
    if 'train' in name:
        return 'train'
    elif 'val' in name:
        return 'val'
    elif 'test' in name:
        return 'test'
    else:
        raise ValueError(f"无法从文件名中识别出数据集类型：{filename}")

def convert_coco_to_yolo_split(coco_json_path, output_dir, images_dir):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    split = get_split_from_filename(coco_json_path)  # train / val / test

    # 创建输出路径
    labels_dir = os.path.join(output_dir, 'labels', split)
    images_out_dir = os.path.join(output_dir, 'images', split)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)

    # 类别映射
    categories = coco['categories']
    # cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    # cat_name_to_index = {name: idx for idx, name in enumerate(sorted(cat_id_to_name.values()))} 
    # cat_id_to_index = {cat['id']: cat_name_to_index[cat['name']] for cat in categories}

    categories_sorted = sorted(categories, key=lambda x: x['id'])  # 按 category_id 排序
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories_sorted}
    cat_id_to_index = {cat['id']: idx for idx, cat in enumerate(categories_sorted)}

    # 写 classes.txt（只写一次）
    # classes_txt_path = os.path.join(output_dir, 'labels', 'classes.txt')
    # if not os.path.exists(classes_txt_path):
    #     with open(classes_txt_path, 'w') as f:
    #         for name in sorted(cat_name_to_index.keys()):
    #             f.write(name + '\n')

    classes_txt_path = os.path.join(output_dir, 'labels', 'classes.txt')
    if not os.path.exists(classes_txt_path):
        with open(classes_txt_path, 'w') as f:
            for cat in categories_sorted:
                f.write(cat['name'] + '\n')

    # 图像信息 & 标注分组
    img_id_to_info = {img['id']: img for img in coco['images']}
    img_id_to_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_annotations:
            img_id_to_annotations[img_id] = []
        img_id_to_annotations[img_id].append(ann)

    for img_id, anns in tqdm(img_id_to_annotations.items(), desc=f"Converting {split}"):
        img_info = img_id_to_info[img_id]
        img_w, img_h = img_info['width'], img_info['height']
        img_filename = img_info['file_name']
        src_img_path = os.path.join(images_dir, img_filename)
        dst_img_path = os.path.join(images_out_dir, os.path.basename(img_filename))

        # 复制图片
        shutil.copy(src_img_path, dst_img_path)

        # 生成 YOLO 标签
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)

        lines = []
        for ann in anns:
            cat_id = ann['category_id']
            bbox = ann['bbox']
            yolo_box = coco_to_yolo_bbox(bbox, img_w, img_h)
            class_id = cat_id_to_index[cat_id]
            lines.append(f"{class_id} " + " ".join(f"{x:.6f}" for x in yolo_box))

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

    print(f"✅ {split} 数据集转换完成！图片与标签已输出到 {images_out_dir} 与 {labels_dir}")

# 示例调用方式
if __name__ == "__main__":
    # 你可以多次调用这个函数，分别处理 train/val/test 三个 json 文件

    phases = ['train', 'val', 'test']
    for pha in phases:
        convert_coco_to_yolo_split(
            coco_json_path=f"/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/{pha}.json",
            output_dir=f"/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH_heart_yolo/c1",
            images_dir=f"/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSH2/Heart/c1/src"  # 所有图片混在一起的原始目录
        )


