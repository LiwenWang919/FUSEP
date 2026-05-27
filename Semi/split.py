import os
import json
import random

def split_coco_dataset(coco_file, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # Load COCO dataset
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Shuffle images
    random.shuffle(images)

    # Calculate number of images for each split
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = int(num_images * test_ratio)

    # Split images
    train_images = images[:num_train]
    val_images = images[num_train:num_train+num_val]
    test_images = images[num_train+num_val:]

    # Collect annotations for each split
    train_annotations = []
    val_annotations = []
    test_annotations = []

    # Build image id set for quick lookup
    train_image_ids = set(image['id'] for image in train_images)
    val_image_ids = set(image['id'] for image in val_images)
    test_image_ids = set(image['id'] for image in test_images)

    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id in train_image_ids:
            train_annotations.append(annotation)
        elif image_id in val_image_ids:
            val_annotations.append(annotation)
        elif image_id in test_image_ids:
            test_annotations.append(annotation)

    # Create new COCO datasets for each split
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data['categories']
    }

    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data['categories']
    }

    test_data = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": coco_data['categories']
    }

    # Save COCO format data for each split
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'val.json')
    test_file = os.path.join(output_dir, 'test.json')

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)

# Example usage
coco_file = '/media/Storage1/wlw/mm/CRL/coco_modified.json'
output_dir = '/media/Storage1/wlw/mm/CRL/'
split_coco_dataset(coco_file, output_dir)