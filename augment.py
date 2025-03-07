import os
import json
import cv2
import argparse
import numpy as np
import albumentations as A

def load_json(json_path):
    """Load a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_json(json_data, json_path):
    """Save a JSON file."""
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def apply_transformation(image, transform):
    """Apply a single transformation to an image."""
    transformed = transform(image=image)
    return transformed['image']

def augment_dataset(input_dir, output_dir):
    """
    Apply data augmentation to a dataset.

    Args:
        input_dir (str): Directory of the dataset to augment.
        output_dir (str): Path to save the augmented dataset.
    """
    # Define augmentation transformations
    transformations = [
        A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.ChannelShuffle(p=0.5),
        A.GaussNoise(std_range=(0.1, 0.2), p=1.0)
    ]
    
    # Locate annotation file
    annotations_path = os.path.join(input_dir, "annotations")
    json_files = [f for f in os.listdir(annotations_path) if f.endswith(".json")]

    if not json_files:
        raise FileNotFoundError(f"No JSON annotation file found in {annotations_path}")

    json_path = os.path.join(annotations_path, json_files[0])
    data = load_json(json_path)

    # Create output directories
    output_images_dir = os.path.join(output_dir, "images/Test")
    output_annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    new_json_path = os.path.join(output_annotations_dir, 'person_keypoints_default.json')

    # Prepare new images & annotations
    new_images = []
    new_annotations = []
    annotation_id = max(ann['id'] for ann in data['annotations']) + 1
    image_id = max(img['id'] for img in data['images']) + 1

    for img in data['images']:
        img_path = os.path.join(input_dir, "images/Test", img['file_name'])
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]

        for idx, transform in enumerate(transformations):
            new_img = apply_transformation(image, transform)

            new_filename = f"aug_{idx}_{img['file_name']}"
            new_img_path = os.path.join(output_images_dir, new_filename)
            cv2.imwrite(new_img_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

            new_images.append({
                "id": image_id,
                "width": new_img.shape[1],
                "height": new_img.shape[0],
                "file_name": new_filename,
                "license": img.get("license", ""),
                "date_captured": img.get("date_captured", "")
            })

            for ann in keypoints:
                new_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "keypoints": ann["keypoints"],
                    "num_keypoints": ann["num_keypoints"]
                })
                annotation_id += 1

            image_id += 1

    # Update the dataset with new images and annotations
    data['images'].extend(new_images)
    data['annotations'].extend(new_annotations)
    save_json(data, new_json_path)

    print(f"Augmentation complete! Processed dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply data augmentation to a COCO keypoint detection dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory (original dataset).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the augmented dataset.")

    args = parser.parse_args()

    augment_dataset(args.input_dir, args.output_dir)
