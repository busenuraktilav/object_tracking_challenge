from ultralytics import YOLO
import cv2
import yaml
from dotmap import DotMap
from pathlib import Path
import numpy as np
from collections import defaultdict
import sys
import os

sys.path.append(os.getcwd())
from src.task_type import Tasks

CONFIG_PATH: Path = Path.cwd().joinpath("config.yaml")

def process_frames(model: YOLO, cap: cv2.VideoCapture, config) -> None:

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(config.output_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (h, w) if config.rotate_video else (w, h))
    
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
            
        if config.task_type == Tasks.SPEED_ESTIMATION.value:
            results = model.track(frame, persist=True, show=False)
            annotated_frame = results[0].plot()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))

                    if len(track) > 1:  
                        dx = track[-1][0] - track[-2][0]
                        dy = track[-1][1] - track[-2][1]
                        distance_pixels = np.sqrt(dx**2 + dy**2)
                        distance_meters = distance_pixels / config.scale

                        time_seconds = 1 / fps
                        speed_m_per_s = distance_meters / time_seconds
                        speed_km_per_h = speed_m_per_s * 3.6

                        cv2.putText(annotated_frame, f"Speed: {int(speed_km_per_h)} km/h", 
                                    (int(x), int(y - h/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (147, 20, 255), 2)

                    if len(track) > 30: 
                        track.pop(0)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            video_writer.write(frame)

        elif config.task_type == Tasks.OBJECT_DETECTION.value:
            results = model(frame) # for detecting all objects
            frame_ = results[0].plot()
            cv2.imshow("frame", frame_)

        elif config.task_type == Tasks.OBJECT_TRACKING.value:
            results = model.track(frame, persist=True) # for only tracking moving objects
            frame_ = results[0].plot()
            cv2.imshow("frame", frame_)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error opening video stream or file"
    return cap

def main() -> None:
    with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as f:
        config = DotMap(yaml.safe_load(f))

    model = YOLO(config.model_name)
    names = model.model.names

    cap = open_video(config.input_video_path)

    process_frames(model, cap, config)

              
        
if __name__ == "__main__":
    main()     

