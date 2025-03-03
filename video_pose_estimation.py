import cv2
import torch
import numpy as np
from ultralytics import YOLO  # YOLOPose Model

# Charger les modèles YOLO
model_detect = YOLO("model/Player-Detection-YOLOv11X-2024-12.pt")
model_pose = YOLO("model/yolo11x-pose.pt")  

def process_video(video_path, output_path):
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Récupérer les propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    padding = 15  # Ajouter un padding autour du crop
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo
        
        # Détection des joueurs avec YOLO (premier modèle)
        results_detect = model_detect(frame)
        
        for result in results_detect:
            for box in result.boxes.xyxy:  # Coordonnées des bounding boxes
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Ajouter le padding et s'assurer que les valeurs restent dans les limites de l'image
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
                
                player_crop = frame[y1:y2, x1:x2]  # Extraire l'image du joueur avec padding
                
                # Pose estimation sur le joueur détecté
                results_pose = model_pose(player_crop)
                
                # Sélectionner le meilleur humain détecté (le plus grand ou avec le plus de keypoints)
                best_pose = None
                max_keypoints = 0
                max_area = 0
                
                for pose_result in results_pose:
                    for pose in pose_result.keypoints.xy:
                        pose = pose.cpu().numpy().astype(int)
                        if pose.size == 0:
                            continue  # Ignorer les poses vides
                        
                        valid_keypoints = np.count_nonzero((pose[:, 0] > 0) & (pose[:, 1] > 0))
                        
                        if valid_keypoints == 0:
                            continue  # Ignorer les poses sans keypoints détectés
                        
                        # Calculer l'aire approximative de l'humain détecté
                        x_min, y_min = np.min(pose[:, 0]), np.min(pose[:, 1])
                        x_max, y_max = np.max(pose[:, 0]), np.max(pose[:, 1])
                        area = (x_max - x_min) * (y_max - y_min)
                        
                        # Sélectionner le meilleur candidat
                        if valid_keypoints > max_keypoints or area > max_area:
                            best_pose = pose
                            max_keypoints = valid_keypoints
                            max_area = area
                
                # Dessiner seulement la meilleure pose détectée
                if best_pose is not None:
                    for x, y in best_pose:
                        if x > 0 and y > 0:  # Éviter les outliers (points non détectés)
                            cv2.circle(frame, (x1 + x, y1 + y), 3, (0, 255, 0), -1)
                    
                    # Dessiner les lignes entre les keypoints (skeleton)
                    skeleton = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (5, 7), (7, 9),
                                (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
                                (13, 15), (12, 14), (14, 16)]
                    for i, j in skeleton:
                        if i < len(best_pose) and j < len(best_pose) and best_pose[i][0] > 0 and best_pose[i][1] > 0 and best_pose[j][0] > 0 and best_pose[j][1] > 0:
                            cv2.line(frame, (x1 + best_pose[i][0], y1 + best_pose[i][1]),
                                     (x1 + best_pose[j][0], y1 + best_pose[j][1]), (255, 0, 0), 2)
        
        # Ajouter la frame modifiée à la vidéo
        out.write(frame)
        
        # # Affichage en temps réel (optionnel)
        # cv2.imshow('YOLOPose Output', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Exécution
if __name__ == "__main__":
    input_video = "video0.mkv"  # Remplace par ta vidéo
    output_video = "output.mp4"
    process_video(input_video, output_video)
    print(f"✅ Vidéo enregistrée : {output_video}")
