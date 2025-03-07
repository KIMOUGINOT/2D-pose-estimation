import os
import json
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip().split() for line in f.readlines()]

def save_txt(txt_data, txt_path):
    with open(txt_path, 'w') as f:
        for line in txt_data:
            f.write(" ".join(map(str, line)) + "\n")

transformations = [
    A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.ChannelShuffle(p=0.5),
    A.GaussNoise(std_range=(0.1, 0.2), p=1.0)
]

def apply_transformation(image, bboxes, transform):
    transformed = transform(image=image, bboxes=bboxes)
    new_image = transformed['image']
    new_bboxes = transformed['bboxes']
    return new_image, new_bboxes

def augment_dataset(input_dir, output_dir):
    """ Apply data augmentation to a YOLO dataset while maintaining its directory structure.

    Args:
        input_dir (str): Directory of the YOLO dataset to augment
        output_dir (str): Path to the augmented YOLO dataset
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Copy the data.yaml file
    data_yaml_path = os.path.join(input_dir, "data.yaml")
    if os.path.exists(data_yaml_path):
        os.system(f"copy {data_yaml_path} {output_dir}")
    
    # Copy the Train.txt file and update paths
    train_txt_path = os.path.join(input_dir, "Train.txt")
    if os.path.exists(train_txt_path):
        with open(train_txt_path, 'r') as f:
            lines = f.readlines()
        
        new_train_txt_path = os.path.join(output_dir, "Train.txt")
        with open(new_train_txt_path, 'w') as f:
            for line in lines:
                line = line.strip()
                filename = os.path.basename(line)
                for idx in range(len(transformations)):
                    new_filename = f"aug_{idx}_{filename}"
                    new_path = os.path.join("data/images/Train/augmented", new_filename)
                    f.write(new_path + "\n")
    
    image_files = [f for f in os.listdir(os.path.join(input_dir, "images/Train/67b8790e4f3cd829917368/images")) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(os.path.join(input_dir, "images/Train/67b8790e4f3cd829917368"), "images", img_file)
        # print(f"img_path = {img_path}")
        label_path = os.path.join(os.path.join(input_dir, "labels/Train/67b8790e4f3cd829917368"), "images", img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        # print(f"label_path = {label_path}")
        image = cv2.imread(img_path)
        if image is None or not os.path.exists(label_path):
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        labels = load_txt(label_path)
        bboxes = [list(map(float, label[1:])) for label in labels]
        
        for idx, transform in enumerate(transformations):
            new_img, new_bboxes = apply_transformation(image, bboxes, transform)
            
            new_filename = f"aug_{idx}_{img_file}"
            new_img_path = os.path.join(images_dir, new_filename)
            cv2.imwrite(new_img_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
            
            new_label_path = os.path.join(labels_dir, new_filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            new_labels = [[label[0]] + bbox for label, bbox in zip(labels, new_bboxes)]
            save_txt(new_labels, new_label_path)

# Exemple d'utilisation
input_dir = "yolo_train_dataset"
output_dir = "yolo_train_dataset_augmented"

augment_dataset(input_dir, output_dir)
