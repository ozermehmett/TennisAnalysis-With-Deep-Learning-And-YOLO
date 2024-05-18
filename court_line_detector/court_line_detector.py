import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2


class CourtLineDetector:
    def __init__(self, model_path):
        """
        Initializes the CourtLineDetector with a pre-trained ResNet50 model adapted for court line detection.

        :param model_path: str
            The file path to the pre-trained model's state dictionary.
        """
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.device_type = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device_type))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        Predicts the key points of court lines in the given image.

        :param image: numpy.ndarray
            The input image in BGR format.

        :return: numpy.ndarray
            The predicted key points as a 1D array with x and y coordinates.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image_tensor)

        key_points = output.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]

        key_points[::2] *= original_w / 224.0
        key_points[1::2] *= original_h / 224.0

        return key_points

    def draw_key_points(self, image, key_points):
        """
        Draws the predicted key points on the given image.

        :param image: numpy.ndarray
            The input image on which to draw the key points.

        :param key_points: numpy.ndarray
            The key points to be drawn as a 1D array with x and y coordinates.

        :return: numpy.ndarray
            The image with key points drawn on it.
        """
        for i in range(0, len(key_points), 2):
            x = key_points[i]
            y = key_points[i+1]
            cv2.putText(image, str(i // 2), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            cv2.circle(image, (int(x), int(y)), 5,
                       (255, 0, 0), -1)

        return image

    def draw_key_points_on_video(self, video_frames, key_points):
        """
        Draws the predicted key points on each frame of the given video.

        :param video_frames: list of numpy.ndarray
            A list of video frames on which to draw the key points.

        :param key_points: numpy.ndarray
            The key points to be drawn on each frame as a 1D array with x and y coordinates.

        :return: list of numpy.ndarray
            The video frames with key points drawn on them.
        """
        output_video_frames = list()
        for frame in video_frames:
            frame = self.draw_key_points(frame, key_points)
            output_video_frames.append(frame)

        return output_video_frames
