import os
import cv2
import argparse
import torch
import numpy as np
from ultralytics import YOLO

# Load the detection model globally (assumed to be constant)
DETECTION_MODEL_PATH = "model/Player-Detection-YOLOv11X-2024-12.pt"
model_detect = YOLO(DETECTION_MODEL_PATH)

def process_video(video_path, pose_model_path):
    """
    Process a video to detect players and estimate their pose.

    Args:
        video_path (str): Path to the input video.
        pose_model_path (str): Path to the pose estimation YOLO model.
    
    Saves:
        A processed video in the same directory with "_output" appended to the filename.
    """
    model_pose = YOLO(pose_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_name, ext = os.path.splitext(video_path)
    output_path = f"{base_name}_output{ext}"

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    padding = 15  # Padding around player crop

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        results_detect = model_detect(frame)

        for result in results_detect:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])

                # Apply padding and ensure values are within image bounds
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)

                player_crop = frame[y1:y2, x1:x2]

                # Run pose estimation
                results_pose = model_pose(player_crop)

                best_pose = select_best_pose(results_pose)

                if best_pose is not None:
                    draw_pose(frame, best_pose, x1, y1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved at: {output_path}")

def select_best_pose(results_pose):
    """
    Select the best detected pose based on the number of keypoints and bounding area.

    Args:
        results_pose: Pose estimation results from YOLO.

    Returns:
        np.ndarray or None: The best detected pose as a set of (x, y) keypoints.
    """
    best_pose = None
    max_keypoints = 0
    max_area = 0

    for pose_result in results_pose:
        for pose in pose_result.keypoints.xy:
            pose = pose.cpu().numpy().astype(int)
            if pose.size == 0:
                continue

            valid_keypoints = np.count_nonzero((pose[:, 0] > 0) & (pose[:, 1] > 0))
            if valid_keypoints == 0:
                continue

            # Compute approximate bounding box area of the pose
            x_min, y_min = np.min(pose[:, 0]), np.min(pose[:, 1])
            x_max, y_max = np.max(pose[:, 0]), np.max(pose[:, 1])
            area = (x_max - x_min) * (y_max - y_min)

            # Select best candidate based on keypoint count or area
            if valid_keypoints > max_keypoints or area > max_area:
                best_pose = pose
                max_keypoints = valid_keypoints
                max_area = area

    return best_pose

def draw_pose(frame, pose, x_offset, y_offset):
    """
    Draw detected pose on the frame with distinct colors for different body parts.

    Args:
        frame (np.ndarray): The original frame.
        pose (np.ndarray): Keypoint coordinates.
        x_offset (int): X offset for drawing.
        y_offset (int): Y offset for drawing.
    """
    colors = {
        "legs": (0, 165, 255),  # Orange
        "core": (0, 255, 255),  # Yellow
        "arms": (0, 255, 0),    # Green
        "head": (255, 0, 0)     # Blue
    }

    skeleton_parts = {
        "legs": [(11, 13), (13, 15), (12, 14), (14, 16)],  # Legs
        "core": [(5, 11), (6, 12), (11, 12)],  # torso
        "arms": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)],  # Arms
        "head": [(0, 1), (1, 2), (2, 3), (3, 4)]  # Head & Neck
    }

    for x, y in pose:
        if x > 0 and y > 0:
            cv2.circle(frame, (x_offset + x, y_offset + y), 3, (255, 255, 255), -1)  # White keypoints

    for part, connections in skeleton_parts.items():
        color = colors[part]
        for i, j in connections:
            if i < len(pose) and j < len(pose) and pose[i][0] > 0 and pose[i][1] > 0 and pose[j][0] > 0 and pose[j][1] > 0:
                cv2.line(frame, (x_offset + pose[i][0], y_offset + pose[i][1]),
                         (x_offset + pose[j][0], y_offset + pose[j][1]), color, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with YOLO pose estimation.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--pose_model", type=str, required=True, help="Path to the pose estimation YOLO model")

    args = parser.parse_args()

    process_video(video_path=args.video, pose_model_path=args.pose_model)
