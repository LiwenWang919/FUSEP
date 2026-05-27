import json
import os

def process_json(json_path, prefix, img_folder):
    """处理单个JSON文件并生成文件名映射"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    name_map = {}
    images = data['images']
    
    # 按顺序生成新文件名
    for idx, img in enumerate(images, start=1):
        old_name = img['file_name']
        new_name = f"{prefix}_{idx:05d}.jpg"
        name_map[old_name] = new_name
        img['file_name'] = new_name  # 更新JSON中的文件名
    
    # 保存修改后的JSON（覆盖原文件，建议先备份）
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return name_map

def rename_images(img_folder, name_map):
    """批量重命名图片文件"""
    for old_name, new_name in name_map.items():
        old_path = os.path.join(img_folder, old_name)
        new_path = os.path.join(img_folder, new_name)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"Warning: {old_path} not found!")

if __name__ == "__main__":
    img_folder = "/media/Storage2/Lvxg/ToMo-UDA/dataset/FUSSD/SA/src"  # 替换为你的图片文件夹路径
    
    # 处理三个JSON文件并获取映射关系
    train_map = process_json("dataset/FUSSD/SA/train.json", "tr", img_folder)
    val_map = process_json("dataset/FUSSD/SA/val.json", "v", img_folder)
    test_map = process_json("dataset/FUSSD/SA/test.json", "te", img_folder)
    
    # 合并所有映射关系
    total_map = {**train_map, **val_map, **test_map}
    
    # 执行文件重命名
    rename_images(img_folder, total_map)