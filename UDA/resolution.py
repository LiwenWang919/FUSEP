import os
from PIL import Image
from collections import defaultdict

def get_image_resolution(image_path):
    """获取图片的分辨率"""
    with Image.open(image_path) as img:
        return img.size  # 返回 (宽度, 高度)

def collect_resolutions(folder_path):
    """遍历文件夹，统计所有图片的分辨率"""
    resolution_counts = defaultdict(int)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 确保是文件而不是子文件夹
        if os.path.isfile(file_path):
            try:
                resolution = get_image_resolution(file_path)
                resolution_counts[resolution] += 1
            except Exception as e:
                print(f"无法处理文件 {filename}: {e}")

    return resolution_counts

def print_statistics(resolution_counts):
    """打印统计信息"""
    print("分辨率统计:")
    for resolution, count in resolution_counts.items():
        print(f"{resolution[0]}x{resolution[1]}: {count} 张图片")

if __name__ == "__main__":
    folder_path = "dataset/abdomen/GE/src"  # 替换为你的文件夹路径
    resolution_counts = collect_resolutions(folder_path)
    print_statistics(resolution_counts)
