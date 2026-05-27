import os
import json
from PIL import Image

def convert_to_coco(json_files, image_folders, output_files):
    global_category_dict = {}
    global_image_id = 1
    global_annotation_id = 1

    for json_file, image_folder, output_file in zip(json_files, image_folders, output_files):
        with open(json_file, 'r') as f:
            data = json.load(f)['annotations']
        
        images = []
        annotations = []

        for image_name, image_data in data.items():
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                print(f"Image '{image_name}' not found. Skipping...")
                continue
            
            # Get image size
            image = Image.open(image_path)
            width, height = image.size

            # Add image information
            image_info = {
                "id": global_image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            }
            images.append(image_info)

            # Add annotations
            for annotation in image_data["annotations"]:
                category_name = annotation["name"]
                if category_name not in global_category_dict:
                    global_category_dict[category_name] = len(global_category_dict) + 1

                category_id = global_category_dict[category_name]
                bbox = get_bbox(annotation["vertex"], width, height)

                annotation_info = {
                    "id": global_annotation_id,
                    "image_id": global_image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                annotations.append(annotation_info)
                global_annotation_id += 1

            global_image_id += 1

        categories = [{"id": cat_id, "name": cat_name} for cat_name, cat_id in global_category_dict.items()]

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        # Save COCO format data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False)

def get_bbox(vertex, width, height):
    x_min = min(vertex[0][0], vertex[1][0])
    y_min = min(vertex[0][1], vertex[1][1])
    x_max = max(vertex[0][0], vertex[1][0])
    y_max = max(vertex[0][1], vertex[1][1])
    return [x_min, y_min, x_max - x_min, y_max - y_min]

# Example usage
json_files = [
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_1/annotations/three_vessel_tracheal_annotations.json",
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_2/annotations/three_vessel_tracheal_annotations.json",
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_3/annotations/three_vessel_tracheal_annotations.json"
]

image_folders = [
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_1/three_vessel_tracheal",
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_2/three_vessel_tracheal",
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_3/three_vessel_tracheal"
]

output_files = [
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_1/annotations/3VT_C1_COCO.json",
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_2/annotations/3VT_C2_COCO.json",
    "/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_3/annotations/3VT_C3_COCO.json"
]

convert_to_coco(json_files, image_folders, output_files)
