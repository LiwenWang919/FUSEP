import json

def merge_coco_files(coco_file1, coco_file2, output_file):
    # Load the two COCO files
    with open(coco_file1, 'r') as f:
        coco_data1 = json.load(f)
    
    with open(coco_file2, 'r') as f:
        coco_data2 = json.load(f)

    # Initialize the merged data
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": coco_data1["categories"]
    }

    # Helper function to get the next available ID
    def get_next_id(data, key):
        if len(data[key]) == 0:
            return 1
        return max(item["id"] for item in data[key]) + 1

    # Merge images and annotations, ensuring unique IDs
    image_id_offset = get_next_id(coco_data1, "images")
    annotation_id_offset = get_next_id(coco_data1, "annotations")

    # Add images and annotations from the first COCO file
    merged_data["images"].extend(coco_data1["images"])
    merged_data["annotations"].extend(coco_data1["annotations"])

    # Add images and annotations from the second COCO file, with ID offsets
    for image in coco_data2["images"]:
        image["id"] += image_id_offset
        merged_data["images"].append(image)

    for annotation in coco_data2["annotations"]:
        annotation["id"] += annotation_id_offset
        annotation["image_id"] += image_id_offset
        merged_data["annotations"].append(annotation)

    # Save the merged COCO file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

# Example usage
coco_file1 = "/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/3VT/c1/train/annotation.json"
coco_file2 = "/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/3VT/c1/val/annotation.json"
output_file = "/media/Storage1/wlw/Semi/SSOD/fetus_annotations_coco/3VT/c1/train/annotation_tv.json"

merge_coco_files(coco_file1, coco_file2, output_file)
