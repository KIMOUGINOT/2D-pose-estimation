import os
import argparse
import torch
from ultralytics import YOLO

def train_yolo(model_name, dataset_path, epochs=50, batch_size=16, img_size=640, device='cuda', freeze=0):
    """
    Train a YOLO model on a given dataset.
    
    Args:
        model_name (str): YOLO model to use (e.g., 'yolov8n.pt', 'yolov5s.pt')
        dataset_path (str): Path to the dataset directory containing data.yaml
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 16.
        img_size (int, optional): Image size. Defaults to 640.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        freese (int, optional): Number of layers to freeze.
    """
    # Check if dataset exists
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
    
    # Load model
    model = YOLO(model_name)
    
    # Train model
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=os.path.join(dataset_path, "runs"),
        name="yolo_training",
        mosaic=0.0,
        scale=0.0,
        pretrained=model_name,
        freeze=freeze,
        lr0=0.001,
        pose=30
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (e.g., 'yolov8n.pt')")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset containing data.yaml")
    parser.add_argument("--freeze", type=int, required=False, help="Number of layers to freeze")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    train_yolo(
        model_name=args.model,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        freeze=args.freeze
    )
