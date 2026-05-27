import os
import json

def clean_coco_files(json_files, image_dir, output_dir):
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # 获取图片文件名的集合
        valid_images = {os.path.basename(img) for img in os.listdir(image_dir)}

        # 筛选存在的图片记录
        valid_image_records = []
        for image in data['images']:
            if os.path.basename(image['file_name']) in valid_images:
                valid_image_records.append(image)

        # 获取有效图像 ID
        valid_image_ids = {img['id'] for img in valid_image_records}

        # 筛选有效的注释记录
        valid_annotations = [ann for ann in data['annotations'] if ann['image_id'] in valid_image_ids]

        # 构建新的数据结构
        new_data = {
            'images': valid_image_records,
            'annotations': valid_annotations,
            'categories': data['categories']  # 保持类别信息
        }

        # 生成输出文件名
        output_file = os.path.join(output_dir, os.path.basename(json_file))

        # 保存到新的 JSON 文件
        with open(output_file, 'w') as f:
            json.dump(new_data, f, indent=4)

        print(f"Processed {json_file}, saved to {output_file}")

# 参数配置
json_files = ['/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/train.json', '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/train_label_1_annotation.json', '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/train_label_5_annotation.json', '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/train_label_10_annotation.json', '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/val.json', '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/test.json']  # 需要处理的 JSON 文件列表
image_dir = '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/'  # 图片目录路径
output_dir = '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/Dataset_For_NIPS_DB_Track/CRL/'  # 输出目录路径

# 清理并保存文件
clean_coco_files(json_files, image_dir, output_dir)
