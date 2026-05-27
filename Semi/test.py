import json

max_bbox_mAP = None

# 逐行读取JSON文件
with open('/media/Storage1/wlw/Semi/SSOD/work_dirs/labelmatch_standard_4-1-10/20240408_134029.log.json') as f:
    for line in f:
        try:
            data = json.loads(line)
            # 检查是否存在 "bbox_mAP" 键
            if "bbox_mAP" in data:
                bbox_mAP = data["bbox_mAP"]
                if max_bbox_mAP is None or bbox_mAP > max_bbox_mAP:
                    max_bbox_mAP = bbox_mAP
        except json.JSONDecodeError:
            print("无法解析行:", line)

if max_bbox_mAP is not None:
    print("最大的 bbox_mAP 数值为:", max_bbox_mAP)
else:
    print("未找到有效的 bbox_mAP 数值")



# import json
# import random

# def split_labeled_unlabeled_coco(input_file, output_labeled_file, output_unlabeled_file, labeled_ratio=0.1):
#     with open(input_file, 'r') as f:
#         coco_data = json.load(f)

#     # 获取图像信息列表
#     images = coco_data['images']

#     # 计算 labeled 和 unlabeled 数据集的大小
#     total_images = len(images)
#     labeled_size = int(total_images * labeled_ratio)
#     unlabeled_size = total_images - labeled_size

#     # 随机打乱图像列表
#     random.shuffle(images)

#     # 划分 labeled 和 unlabeled 数据集
#     labeled_images = images[:labeled_size]
#     unlabeled_images = images[labeled_size:]

#     # 从 coco_data 中获取对应的 annotations
#     labeled_annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in [image['id'] for image in labeled_images]]
#     unlabeled_annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in [image['id'] for image in unlabeled_images]]

#     # 构建 labeled 和 unlabeled 数据集的字典
#     labeled_data = {"categories": [
#         {
#             "id": 1,
#             "name": "thalami"
#         },
#         {
#             "id": 2,
#             "name": "nasal bone"
#         },
#         {
#             "id": 3,
#             "name": "palate"
#         },
#         {
#             "id": 4,
#             "name": "nasal skin"
#         },
#         {
#             "id": 5,
#             "name": "nasal tip"
#         },
#         {
#             "id": 6,
#             "name": "midbrain"
#         },
#         {
#             "id": 7,
#             "name": "NT"
#         },
#         {
#             "id": 8,
#             "name": "IT"
#         },
#         {
#             "id": 9,
#             "name": "CM"
#         }
#     ],'images': labeled_images, 'annotations': labeled_annotations}
#     unlabeled_data = {"categories": [
#         {
#             "id": 1,
#             "name": "thalami"
#         },
#         {
#             "id": 2,
#             "name": "nasal bone"
#         },
#         {
#             "id": 3,
#             "name": "palate"
#         },
#         {
#             "id": 4,
#             "name": "nasal skin"
#         },
#         {
#             "id": 5,
#             "name": "nasal tip"
#         },
#         {
#             "id": 6,
#             "name": "midbrain"
#         },
#         {
#             "id": 7,
#             "name": "NT"
#         },
#         {
#             "id": 8,
#             "name": "IT"
#         },
#         {
#             "id": 9,
#             "name": "CM"
#         }
#     ],'images': unlabeled_images, 'annotations': unlabeled_annotations}

#     # 写入 labeled 和 unlabeled 数据集到输出文件
#     with open(output_labeled_file, 'w') as f:
#         json.dump(labeled_data, f)

#     with open(output_unlabeled_file, 'w') as f:
#         json.dump(unlabeled_data, f)

# # Specify input JSON file and output file paths
# input_file = '/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/EP/c1/train/annotation.json'
# output_labeled_file_10 = 'fetus_annotations_coco/EP/c1/train/semi/train_label_10_annotation.json'
# output_unlabeled_file_10 = 'fetus_annotations_coco/EP/c1/train/semi/train_unlabel_10_annotation.json'
# output_labeled_file_5 = 'fetus_annotations_coco/EP/c1/train/semi/train_label_5_annotation.json'
# output_unlabeled_file_5 = 'fetus_annotations_coco/EP/c1/train/semi/train_unlabel_5_annotation.json'
# output_labeled_file_1 = 'fetus_annotations_coco/EP/c1/train/semi/train_label_1_annotation.json'
# output_unlabeled_file_1 = 'fetus_annotations_coco/EP/c1/train/semi/train_unlabel_1_annotation.json'

# # Split the dataset
# split_labeled_unlabeled_coco(input_file, output_labeled_file_10, output_unlabeled_file_10, 0.1)
# split_labeled_unlabeled_coco(input_file, output_labeled_file_5, output_unlabeled_file_5,0.05)
# split_labeled_unlabeled_coco(input_file, output_labeled_file_1, output_unlabeled_file_1, 0.01)




# import csv
# import json
# import os

# import cv2

# def get_image_size_opencv(image_path):
#     # 使用OpenCV读取图像
#     img = cv2.imread(image_path)
#     if img is None:
#         return None, None
#     height, width, _ = img.shape
#     return width, height

# def get_image_size(image_folder, fname):
#     # 获取图像的真实长宽
#     image_path = os.path.join(image_folder, fname)
#     width, height = 0, 0
#     if os.path.exists(image_path):
#         width, height = get_image_size_opencv(image_path)  # 示例，假设图像大小为100x100
#         # 实际应用中，您需要使用适当的库获取图像的真实大小
#     return width, height

# def csv_to_coco(csv_file, image_folder, output_json):
#     coco_data = {
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }
    
#     fname_to_image_info = {}  # 用于处理重复的文件名
    
#     # 读取CSV文件
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             fname = row["fname"]
#             if fname in fname_to_image_info:
#                 image_info = fname_to_image_info[fname]
#             else:
#                 width, height = get_image_size(image_folder, fname)
#                 image_info = {
#                     "id": len(coco_data["images"]) + 1,
#                     "file_name": fname,
#                     "width": width,
#                     "height": height
#                 }
#                 fname_to_image_info[fname] = image_info
#                 coco_data["images"].append(image_info)
#             if row['structure'] == 'thalami':
#                 c = 1
#             if row['structure'] == 'nasal bone':
#                 c = 2
#             if row['structure'] == 'palate':
#                 c = 3
#             if row['structure'] == 'nasal skin':
#                 c = 4
#             if row['structure'] == 'nasal tip':
#                 c = 5
#             if row['structure'] == 'midbrain':
#                 c = 6
#             if row['structure'] == 'NT':
#                 c = 7
#             if row['structure'] == 'IT':
#                 c = 8
#             if row['structure'] == 'CM':
#                 c = 9
            
#             # 添加标注信息
#             annotation = {
#                 "id": len(coco_data["annotations"]) + 1,
#                 "image_id": image_info["id"],
#                 "category_id": c,  # 假设只有一个类别
#                 "bbox": [
#                     float(row["w_min"]),
#                     float(row["h_min"]),
#                     float(row["w_max"]) - float(row["w_min"]),
#                     float(row["h_max"]) - float(row["h_min"])
#                 ],
#                 "iscrowd": 0  # 假设不存在重叠目标
#             }
#             annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
#             coco_data["annotations"].append(annotation)
    
#     # 添加类别信息（可选）
#     coco_data["categories"].append({"id": 1, "name": "thalami"})
#     coco_data["categories"].append({"id": 2, "name": "nasal bone"})
#     coco_data["categories"].append({"id": 3, "name": "palate"})
#     coco_data["categories"].append({"id": 4, "name": "nasal skin"})
#     coco_data["categories"].append({"id": 5, "name": "nasal tip"})
#     coco_data["categories"].append({"id": 6, "name": "midbrain"})
#     coco_data["categories"].append({"id": 7, "name": "NT"})
#     coco_data["categories"].append({"id": 8, "name": "IT"})
#     coco_data["categories"].append({"id": 9, "name": "CM"})
    
#     # 保存COCO格式数据为JSON文件
#     with open(output_json, 'w') as f:
#         json.dump(coco_data, f)

# # 指定输入CSV文件和输出JSON文件路径
# csv_file = '/media/Storage1/wlw/Semi/SSOD/ObjectDetection.csv'  # 输入CSV文件路径
# output_file = 'output.json'  # 输出JSON文件路径
# IMG_PATH = '/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/EP'

# # 将CSV转换为COCO JSON
# csv_to_coco(csv_file, IMG_PATH, output_file)



# import json
# import random
# from collections import defaultdict

# def split_coco(coco_json, train_ratio, val_ratio, test_ratio, output_dir):
#     # 读取原始COCO数据
#     with open(coco_json, 'r') as f:
#         coco_data = json.load(f)
    
#     # 计算每个集合的数量
#     total_images = len(coco_data["images"])
#     train_count = int(total_images * train_ratio)
#     val_count = int(total_images * val_ratio)
#     test_count = int(total_images * test_ratio)
    
#     # 随机选择图片ID
#     image_ids = [image["id"] for image in coco_data["images"]]
#     random.shuffle(image_ids)
    
#     # 分配图片到各个集合
#     train_images = image_ids[:train_count]
#     val_images = image_ids[train_count:train_count+val_count]
#     test_images = image_ids[train_count+val_count:train_count+val_count+test_count]
    
#     # 构建分割后的COCO数据
#     split_coco_data = defaultdict(lambda: {"images": [], "annotations": [], "categories": coco_data["categories"]})
#     for image in coco_data["images"]:
#         if image["id"] in train_images:
#             split_coco_data["train"]["images"].append(image)
#         elif image["id"] in val_images:
#             split_coco_data["val"]["images"].append(image)
#         elif image["id"] in test_images:
#             split_coco_data["test"]["images"].append(image)
    
#     # 同时处理对应的标注信息
#     for annotation in coco_data["annotations"]:
#         image_id = annotation["image_id"]
#         if image_id in train_images:
#             split_coco_data["train"]["annotations"].append(annotation)
#         elif image_id in val_images:
#             split_coco_data["val"]["annotations"].append(annotation)
#         elif image_id in test_images:
#             split_coco_data["test"]["annotations"].append(annotation)
    
#     # 保存分割后的COCO数据
#     for split, data in split_coco_data.items():
#         with open(f"{output_dir}/instances_{split}.json", 'w') as f:
#             json.dump(data, f)

# # 示例用法
# split_coco("/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/EP/c1/output.json", 0.7, 0.1, 0.2, "/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/EP/c1")