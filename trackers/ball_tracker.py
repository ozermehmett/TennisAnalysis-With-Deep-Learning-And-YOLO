from ultralytics import YOLO
import torch
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device_type = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device_type)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = list()

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = dict()
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_boxes(self, video_frame, ball_detections):
        output_video_frames = list()
        for frame, ball_dict in zip(video_frame, ball_detections):
            # Draw bounding boxes
            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = box

                cv2.putText(frame, f"Ball ID: {track_id}", (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.rectangle(frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              color=(0, 255, 0), thickness=2)

            output_video_frames.append(frame)

        return output_video_frames
