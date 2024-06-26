from ultralytics import YOLO
import torch
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_box


class PlayerTracker:
    def __init__(self, model_path):
        """
        Initializes the PlayerTracker class.

        :param model_path: str
            Path to the YOLO model.
        """
        self.model = YOLO(model_path)
        self.device_type = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device_type)
        self.model.to(self.device_type)

    def choose_and_filter_players(self, court_key_points, player_detections):
        """
        Chooses players based on their proximity to court key points and filters the detections accordingly.

        :param court_key_points: list
            The List of court key points.

        :param player_detections: list
            The List of player detections for each frame.

        :return: list
            The List of filtered player detections for each frame.
        """
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_key_points, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if
                                    track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_key_points, player_dict):
        """
        Chooses players based on their distance to court key points.

        :param court_key_points: list
            The List of court key points.

        :param player_dict: dict
            Dictionary of player detections for a single frame.

        :return: list
            The List of chosen player track IDs.
        """
        distances = list()
        for track_id, box in player_dict.items():
            player_center = get_center_of_box(box)

            min_distance = float('inf')
            for i in range(0, len(court_key_points), 2):
                court_keypoint = (court_key_points[i], court_key_points[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

            # sorrt the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects player positions in a list of video frames.

        :param frames: list
            The List of video frames.

        :param read_from_stub: bool, optional
            Flag to read detections from a stub file.

        :param stub_path: str, optional
            Path to the stub file for reading/writing detections.

        :return: list
            The List of player detections for each frame.
        """
        players_detections = list()

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                players_detections = pickle.load(f)
            return players_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            players_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(players_detections, f)

        return players_detections

    def detect_frame(self, frame):
        """
        Detects player positions in a single video frame.

        :param frame: np.ndarray
            A single video frame.

        :return: dict
            A dictionary containing player detection coordinates with track IDs.
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_boxes(self, video_frame, players_detections):
        """
        Draws bounding boxes around detected players in each frame.

        :param video_frame: list
            The List of video frames.

        :param players_detections: list
            The List of player detection dictionaries for each frame.

        :return: list
            The List of video frames with drawn bounding boxes.
        """
        output_video_frames = list()
        for frame, player_dict in zip(video_frame, players_detections):
            # Draw bounding boxes
            for track_id, box in player_dict.items():
                x1, y1, x2, y2 = box

                cv2.putText(frame, f"Player ID: {track_id}", (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255))

                cv2.rectangle(frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              color=(0, 0, 255), thickness=2)

            output_video_frames.append(frame)

        return output_video_frames
