import os
import shutil
import random

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Les proportions doivent totaliser 1.0"
    
    # Créer les répertoires de sortie
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Récupérer toutes les images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)  # Mélanger les images
    
    # Calculer les tailles des ensembles
    total_images = len(image_files)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    
    # Assigner les fichiers aux ensembles
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    def move_files(file_list, split):
        for img_file in file_list:
            img_src = os.path.join(image_dir, img_file)
            img_dst = os.path.join(output_dir, 'images', split, img_file)
            shutil.move(img_src, img_dst)
            
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_src = os.path.join(label_dir, label_file)
            label_dst = os.path.join(output_dir, 'labels', split, label_file)
            
            if os.path.exists(label_src):
                shutil.move(label_src, label_dst)
    
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')
    
    print(f"Dataset réparti : {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

# Exemple d'utilisation
split_dataset("dataset/yolo_dataset/images/train", "dataset/yolo_dataset/labels/train", "dataset/yolo_dataset", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)