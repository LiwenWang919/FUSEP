# import json

# # 1. 加载JSON文件
# def load_json(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     return data

# # 2. 统计图片数量
# def count_images(data):
#     return len(data['images'])

# # 3. 统计每张图片的平均标签数量
# def average_labels_per_image(data):
#     total_labels = sum(len(img.get('annotations', [])) for img in data['images'])
#     total_images = len(data['images'])
#     return total_labels / total_images

# # 4. 统计每个分类的数量
# def count_categories(data):
#     categories = {}
#     for ann in data.get('annotations', []):
#         category_id = ann.get('category_id')
#         if category_id is not None:
#             if category_id not in categories:
#                 categories[category_id] = 0
#             categories[category_id] += 1
#     return categories

# if __name__ == "__main__":
#     # 假设你有三个COCO的JSON文件：coco1.json、coco2.json和coco3.json
#     json_files = ["/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/4c/c1/train/annotation.json", "/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/4c/c1/val/annotation.json", "/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/4c/c1/test/annotation.json"]
    
#     for json_file in json_files:
#         coco_data = load_json(json_file)
#         num_images = count_images(coco_data)
#         avg_labels_per_image = average_labels_per_image(coco_data)
#         categories_count = count_categories(coco_data)
        
#         print(f"文件: {json_file}")
#         print(f"图像数量: {num_images}")
#         print(f"每张图像的平均标签数量: {avg_labels_per_image}")
#         print("每个分类的数量:")
#         for category_id, count in categories_count.items():
#             print(f"分类ID: {category_id}, 数量: {count}")
#         print("\n")


import json

# 1. 加载JSON文件
def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# 2. 统计图片数量
def count_images(data):
    print(len(data['images']))
    return len(data['images'])

# 3. 统计每张图片的标签数量
def count_labels_per_image(data):
    return sum(len(img.get('annotations', [])) for img in data['images'])

# 4. 统计每个分类的数量
def count_categories(data):
    categories = {}
    for ann in data.get('annotations', []):
        category_id = ann.get('category_id')
        if category_id is not None:
            if category_id not in categories:
                categories[category_id] = 0
            categories[category_id] += 1
    return categories

if __name__ == "__main__":
    total_images = 0
    total_labels = 0
    total_categories = {}
    
    # 假设你有三个COCO的JSON文件：coco1.json、coco2.json和coco3.json
    json_files = ["/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/EP/c1/output.json"]
    
    for json_file in json_files:
        coco_data = load_json(json_file)
        total_images += count_images(coco_data)
        total_labels += count_labels_per_image(coco_data)
        
        categories_count = count_categories(coco_data)
        for category_id, count in categories_count.items():
            if category_id not in total_categories:
                total_categories[category_id] = 0
            total_categories[category_id] += count
        
    print("总体统计信息:")
    print(f"总图像数量: {total_images}")
    
    print("总分类数量:")
    sum = 0
    for category_id, count in total_categories.items():
        sum += count
        # print(f"分类ID: {category_id}, 总数量: {count}")

    print(f"总标签数量: {sum}")

    for category_id, count in total_categories.items():
        print(f"分类ID: {category_id}, 总数量: {count,count/sum}")
