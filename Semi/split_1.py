import json
import random
import os

def extract_samples(input_file, output_dir, percentages):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    image_id_to_annotations = {}
    for ann in annotations:
        image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    image_ids = [img['id'] for img in images]
    total_images = len(image_ids)

    for p in percentages:
        sample_count = int(total_images * p / 100)
        sampled_image_ids = set(random.sample(image_ids, sample_count))
        unsampled_image_ids = set(image_ids) - sampled_image_ids

        # 已标记数据
        labeled_images = [img for img in images if img['id'] in sampled_image_ids]
        labeled_annotations = [ann for img_id in sampled_image_ids for ann in image_id_to_annotations.get(img_id, [])]

        # 未标记数据（包含注释）
        unlabeled_images = [img for img in images if img['id'] in unsampled_image_ids]
        unlabeled_annotations = [ann for img_id in unsampled_image_ids for ann in image_id_to_annotations.get(img_id, [])]

        labeled_data = {
            'images': labeled_images,
            'annotations': labeled_annotations,
            'categories': categories
        }

        unlabeled_data = {
            'images': unlabeled_images,
            'annotations': unlabeled_annotations,
            'categories': categories
        }

        # 输出路径
        labeled_path = os.path.join(output_dir, f'NT_train_label_{p}.json')
        unlabeled_path = os.path.join(output_dir, f'NT_train_unlabel_{p}.json')

        with open(labeled_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, indent=4, ensure_ascii=False)
        with open(unlabeled_path, 'w', encoding='utf-8') as f:
            json.dump(unlabeled_data, f, indent=4, ensure_ascii=False)

# 参数配置
input_file = '/media/Storage3/wlw/NIPSDB/SCSZ/YN-tuomin/Merge/NT_train.json'
output_dir = '/media/Storage3/wlw/NIPSDB/SCSZ/YN-tuomin/Merge/'
extract_samples(input_file, output_dir, [5, 10])
