import cv2


def read_video(path):
    """
    Read a video from the given file path.

    :param path: str
        Path to the video file.

    :return: list
        The List of video frames.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Save a list of video frames as a video file.

    :param output_video_frames: list
        The List of video frames to be saved.

    :param output_video_path: str
        Path where the output video will be saved.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30,
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved the: {output_video_path}")
