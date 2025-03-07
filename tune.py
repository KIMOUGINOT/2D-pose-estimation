import os
import argparse
import torch
from ultralytics import YOLO

def tune_yolo(model_name, dataset_path, epochs=50, iterations=100):
    """
    Tune hyper parameters for a YOLO model on a given dataset.
    
    Args:
        model_name (str): YOLO model to use (e.g., 'yolov8n.pt', 'yolov5s.pt')
        dataset_path (str): Path to the dataset directory containing data.yaml
        epochs (int, optional): Number of training epochs. 
        iterations (int, optional): Number of tuning iterations.
    """
    # Check if dataset exists
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
    
    # Load model
    model = YOLO(model_name)
    
    # Tune model
    model.tune(
        data=data_yaml,
        epochs=epochs,
        iterations=iterations,
        project=os.path.join(dataset_path, "parameter_tuning"),
        name="Results",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (e.g., 'yolov8n.pt')")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset containing data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    
    args = parser.parse_args()
    
    tune_yolo(
        model_name=args.model,
        dataset_path=args.dataset,
        epochs=args.epochs,
        iterations=args.iterations
    )