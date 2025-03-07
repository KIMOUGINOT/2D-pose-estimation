import os

def remove_last_keypoint_from_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Vérifie qu'il s'agit d'un fichier texte
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            with open(file_path, 'w') as file:
                for line in lines:
                    values = line.strip().split()
                    if len(values) >= 3:  # Vérifie qu'il y a assez d'éléments pour supprimer un keypoint
                        new_line = ' '.join(values[:-3])  # Supprime les 3 derniers nombres
                        file.write(new_line + '\n')
    print("finished")

if __name__ == "__main__":
    
    #remove_last_keypoint_from_directory('yolo_dataset/labels/train/67b8790e4f3cd829917368/images')
    remove_last_keypoint_from_directory('yolo_dataset/labels/train/')
    remove_last_keypoint_from_directory('yolo_dataset/labels/val/')
    remove_last_keypoint_from_directory('yolo_dataset/labels/test/')
    #remove_last_keypoint_from_directory('yolo_dataset/labels/val')
