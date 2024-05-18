from ultralytics import YOLO
import torch
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        """
        Initializes the BallTracker class.

        :param model_path: str
            Path to the YOLO model.
        """
        self.model = YOLO(model_path)
        self.device_type = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device_type)

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions.

        :param ball_positions: list
            List of ball positions where each position is a dictionary with key 1.

        :return: list
            List of interpolated ball positions.
        """
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    import pandas as pd

    def get_ball_shot_frames(self, ball_positions):
        """
        Identifies frames where the ball is hit based on its position changes.

        :param ball_positions: list
            List of ball positions where each position is a dictionary with key 1.

        :return: list
            List of frame numbers where the ball is hit.
        """
        ball_positions = [x.get(1, []) for x in ball_positions]

        # Veriyi pandas dataframe'e çevirme
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0

        # Orta noktayı ve farkları hesaplama
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions.loc[i, 'delta_y'] > 0 > df_ball_positions.loc[i + 1, 'delta_y']
            positive_position_change = df_ball_positions.loc[i, 'delta_y'] < 0 < df_ball_positions.loc[i + 1, 'delta_y']

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    if change_frame >= len(df_ball_positions):
                        break
                    negative_position_change_following_frame = (
                            df_ball_positions.loc[i, 'delta_y'] > 0 > df_ball_positions.loc[change_frame, 'delta_y'])
                    positive_position_change_following_frame = (
                            df_ball_positions.loc[i, 'delta_y'] < 0 < df_ball_positions.loc[change_frame, 'delta_y'])

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects ball positions in a list of video frames.

        :param frames: list
            List of video frames.

        :param read_from_stub: bool, optional
            Flag to read detections from a stub file.

        :param stub_path: str, optional
            Path to the stub file for reading/writing detections.

        :return: list
            List of ball detections for each frame.
        """
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
        """
        Detects ball positions in a single video frame.

        :param frame: np.ndarray
            A single video frame.

        :return: dict
            A dictionary containing the ball detection coordinates.
        """
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = dict()
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_boxes(self, video_frame, ball_detections):
        """
        Draws bounding boxes around detected balls in each frame.

        :param video_frame: list
            List of video frames.

        :param ball_detections: list
            List of ball detection dictionaries for each frame.

        :return: list
            List of video frames with drawn bounding boxes.
        """
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
