import os
from tqdm import tqdm

def rename_images_and_labels(image_dir, label_dir, image_prefix):
    image_extensions = {'.jpg', '.jpeg', '.png'}  # Extensions d'images prises en charge
    rename_dict = {}  # Dictionnaire de correspondance ancien -> nouveau nom
    
    # Lister les images et les trier pour assurer un ordre stable
    images = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    
    for idx, image_name in tqdm(enumerate(images)):
        old_image_path = os.path.join(image_dir, image_name)
        new_image_name = f"{image_prefix}_{idx+1}{os.path.splitext(image_name)[1].lower()}"
        new_image_path = os.path.join(image_dir, new_image_name)
        
        # Renommer l'image
        os.rename(old_image_path, new_image_path)
        
        # Renommer le fichier label correspondant si existant
        old_label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
        new_label_path = os.path.join(label_dir, os.path.splitext(new_image_name)[0] + ".txt")
        
        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)

    print("Job done.")



# Exemple d'utilisation
rename_dict = rename_images_and_labels("dataset/yolo_dataset/images/train/sess27_1/images", "dataset/yolo_dataset/labels/train/sess27_1/images", "sess27")