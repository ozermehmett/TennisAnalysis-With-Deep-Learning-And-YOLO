from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2


def main():
    # Read Video
    video_path = "input_videos/input_video.mp4"
    video_frames = read_video(video_path)

    # detect players and ball
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    print(player_tracker.device_type)

    # read_from_stub --> First, set it to False and run it, then set it to True. Otherwise, it will give an error
    player_detections = player_tracker.detect_frames(frames=video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')

    ball_detections = ball_tracker.detect_frames(frames=video_frames,
                                                 read_from_stub=True,
                                                 stub_path='tracker_stubs/ball_detections.pkl')

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector Model
    court_model_path = 'models/key_points_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_key_points = court_line_detector.predict(video_frames[0])

    # choose player
    player_detections = player_tracker.choose_and_filter_players(court_key_points, player_detections)

    # draw detections
    output_video_frames = player_tracker.draw_boxes(video_frame=video_frames,
                                                    players_detections=player_detections)
    output_video_frames = ball_tracker.draw_boxes(video_frame=video_frames,
                                                  ball_detections=ball_detections)

    # draw court key points
    output_video_frames = court_line_detector.draw_key_points_on_video(video_frames=video_frames,
                                                                       key_points=court_key_points)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # save video
    output_video_path = "output_videos/output_video.mp4"
    save_video(output_video_frames=output_video_frames,
               output_video_path=output_video_path)


if __name__ == "__main__":
    main()
