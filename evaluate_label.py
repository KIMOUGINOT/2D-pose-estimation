from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
import json
import os
import cv2

def yolo_to_coco_results(yolo_labels_dir, groundtruth_path, output_coco_results):
    """
    Convertit les prédictions YOLO en format de résultats COCO pour l'évaluation avec pycocotools.
    
    Arguments:
        yolo_labels_dir (str): Dossier contenant les fichiers de prédictions YOLO (.txt).
        groundtruth_path (str): Fichier JSON COCO des annotations ground truth.
        output_coco_results (str): Chemin du fichier JSON de sortie en format COCO.
    """
    

    with open(groundtruth_path, "r") as f:
        coco_gt = json.load(f)

    # Associate files name to image_id COCO
    image_id_map = {img["file_name"]: img["id"] for img in coco_gt["images"]}

    coco_results = []

    for filename in os.listdir(yolo_labels_dir):
        if not filename.endswith(".txt"):
            continue  
        
        image_name = filename.replace(".txt", ".jpg")  
        if image_name not in image_id_map:
            print(f"Image {image_name} ignored (not found in ground truth file).")
            continue
        
        image_id = image_id_map[image_name]
        file_path = os.path.join(yolo_labels_dir, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()
            if len(values) < 2 + 3 * 17:  # 1 class_id + 2 bbox + 3 * 17 keypoints
                print(f"Ignored line in {filename}, incorrect format")
                continue

            class_id = int(values[0]) 
            keypoints = list(map(float, values[2:]))  

            keypoints_abs = []
            # Create the dictionnary
            image_info_map = {img["id"]: img for img in coco_gt["images"]}

            if image_id not in image_info_map:
                print(f"Error : image_id {image_id} not found in the ground truth file.")
                continue 

            width, height = image_info_map[image_id]["width"], image_info_map[image_id]["height"]

            for i in range(1,18):  # 17 keypoints
                x = keypoints[i * 3] * width
                y = keypoints[i * 3 + 1] * height
                visibility = int(round(keypoints[i * 3 + 2]*2))  #Normally 0/1 in YOLO so *2 to get 0/2
                keypoints_abs.extend([x, y, visibility])

            coco_results.append({
                "image_id": image_id,
                "category_id": class_id + 1, 
                "keypoints": keypoints_abs,
                "score": 0.9 
            })

    # Save coco results
    with open(output_coco_results, "w") as f:
        json.dump(coco_results, f, indent=4)

    print(f"COCO file generated : {output_coco_results}")


def setup_detectron2_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Seuil de détection
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Utilisation de GPU si dispo
    return DefaultPredictor(cfg)

def detectron2_to_coco(input_dir, output_json, groundtruth_path):
    predictor = setup_detectron2_model()
    
    with open(groundtruth_path, "r") as f:
        coco_gt = json.load(f)

    # Associate files name to image_id COCO
    image_id_map = {img["file_name"]: img["id"] for img in coco_gt["images"]}

    # Initialisation du fichier COCO
    coco_output = []
    
        # "info": {
        #     "description": "Detectron2 Pose Estimation",
        #     "version": "1.0",
        #     "year": 2024
        # },
        # "licenses": [],
        # "images": [],
        # "annotations": [],
        # "categories": [
        #     {
        #         "id": 1,
        #         "name": "person",
        #         "supercategory": "person",
        #         "keypoints": [
        #             "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
        #             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        #             "left_wrist", "right_wrist", "left_hip", "right_hip",
        #             "left_knee", "right_knee", "left_ankle", "right_ankle"
        #         ],
        #         "skeleton": [
        #             [6, 12], [2, 4], [1, 2], [13, 7], [3, 1], [15, 17], 
        #             [6, 8], [12, 13], [8, 10], [4, 6], [5, 7], [7, 9], 
        #             [5, 3], [13, 15], [7, 6], [14, 12], [9, 11], [16, 14]
        #         ]
        #     }
        # ]

    annotation_id = 1  # ID unique des annotations
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".jpg", ".png")):
            continue  # Ignorer les fichiers non-images

        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image_id = image_id_map[filename]

        # Prédiction des keypoints avec Detectron2
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        for i in range(len(instances)):
            keypoints = instances.pred_keypoints[i].numpy().astype(float) if instances.has("pred_keypoints") else None
            score = float(instances.scores[i]) if instances.has("scores") else 0.0
            bbox = instances.pred_boxes[i].tensor.numpy()[0].tolist() if instances.has("pred_boxes") else [0, 0, 0, 0]

            if keypoints is None:
                print("pas de keypoint trouvé")
                continue  # Ignorer si pas de keypoints détectés

            # Conversion des keypoints en format COCO (x, y, visibility)
            keypoints_coco = []
            num_keypoints = 0
            for x, y, v in keypoints:
                if v > 0:  # Keypoint détecté
                    keypoints_coco.extend([float(x), float(y), int(2)])  # 2 = visible
                    num_keypoints += 1
                else:
                    keypoints_coco.extend([float(0), float(0), int(0)])  # 0 = non visible

            coco_output.append({
                "image_id": int(image_id),
                "category_id": int(annotation_id), 
                "keypoints": keypoints_coco,
                "score": float(score) 
            })


    # Sauvegarde du fichier COCO JSON
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"✅ JSON COCO généré : {output_json}")




def evaluate_label(groundtruth_json, predicted_json):
    """Evaluate the prediction file according to the groundtruth file

    Args:
        groundtruth_json (str): Path to the groundtruth COCO format file.
        predicted_json (str): Path to the prediction COCO format file.
    """

    coco_gt = COCO(groundtruth_json)

    # modify gt
    for ann in coco_gt.anns.values():  
        if len(ann["keypoints"]) == 18 * 3:
            ann["keypoints"] = ann["keypoints"][:17 * 3]  # take off 18th keypoint

    coco_dt = coco_gt.loadRes(predicted_json)

    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



# if __name__ == "__main__" :
#     from ultralytics import YOLO

#     # model = YOLO("./model/yolo11n-pose.pt")
#     model = YOLO("./model/yolo11x-pose.pt")
#     model.predict(source="test_dataset/images/Test", verbose=True, save=True, save_txt=True, project="evaluate_model")
#     yolo_to_coco_results("evaluate_model/predict/labels", "test_dataset/annotations/person_keypoints_test.json","evaluate_model/predictions.json")
#     evaluate_label("test_dataset/annotations/person_keypoints_Test.json","evaluate_model/predictions.json")

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # Seuil de détection
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Utilisation de GPU si dispo
    predictor =  DefaultPredictor(cfg)

    # 📌 2. Charger le dossier d'images
    image_dir = "./test_dataset/images/Test"  # Ex: "./images"
    output_dir = "./detectron2_predictions"  # Où sauvegarder les images annotées
    os.makedirs(output_dir, exist_ok=True)

    # 📌 3. Boucle sur chaque image
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, image_name)
        image = cv2.imread(img_path)

        # 📌 4. Faire des prédictions avec Detectron2
        outputs = predictor(image)

        # 📌 5. Visualiser les annotations
        v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 📌 7. Sauvegarder l'image annotée
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])

    cv2.destroyAllWindows()
    print(f"✅ Images annotées enregistrées dans {output_dir}")
    
    input_dir = "./test_dataset/images/Test"  # Dossier contenant les images
    output_json = "./detectron2_predictions/detectron2_predictions.json"
    groundtruth_dir = "./test_dataset/annotations/person_keypoints_Test.json"
    detectron2_to_coco(input_dir, output_json, groundtruth_dir)
    evaluate_label("test_dataset/annotations/person_keypoints_Test.json","./detectron2_predictions.json")
