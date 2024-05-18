import cv2
import sys
import numpy as np
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

sys.path.append('../')


class MiniCourt:
    def __init__(self, frame):
        """
        Initializes the MiniCourt with the given frame.

        :param frame: numpy.ndarray
            The frame from which the mini court will be initialized.
        """
        self.draw_rectangle_width = 250
        self.draw_rectangle_height = 450
        self.buffer = 50
        self.padding_court = 20

        # Set canvas background positions
        self.end_x, self.end_y, self.start_x, self.start_y = self.set_canvas_background_box_position(frame)

        # Mini court positions
        self.court_start_x, self.court_start_y, self.court_end_x, self.court_end_y, self.court_drawing_width = self.set_mini_court_position()

        self.drawing_key_points = self.set_court_drawing_key_points()

        self.lines = self.set_court_lines()

    def set_court_lines(self):
        """
        Defines the lines that make up the court.

        :return: list of tuples
            Each tuple contains pairs of indices representing the start and end points of each line.
        """
        return [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def set_mini_court_position(self):
        """
        Sets the positions and dimensions of the mini court.

        :return: tuple
            The start and end coordinates and the drawing width of the mini court.
        """
        court_start_x = self.start_x + self.padding_court
        court_start_y = self.start_y + self.padding_court
        court_end_x = self.end_x - self.padding_court
        court_end_y = self.end_y - self.padding_court
        court_drawing_width = court_end_x - court_start_x

        return court_start_x, court_start_y, court_end_x, court_end_y, court_drawing_width

    def convert_meters_to_pixels(self, meters):
        """
        Converts a distance from meters to pixels.

        :param meters: float
            The distance in meters to be converted.

        :return: float
            The distance converted to pixels.
        """
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                                )

    def set_court_drawing_key_points(self):
        """
        Sets the key points for drawing the court.

        :return: list of int
            A list of key points' coordinates for drawing the court.
        """
        drawing_key_points = [0] * 28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        return drawing_key_points

    def set_canvas_background_box_position(self, frame):
        """
        Sets the position of the canvas background box based on the frame.

        :param frame: numpy.ndarray
            The input frame to determine the background box position.

        :return: tuple
            The end and start coordinates of the background box.
        """
        frame = frame.copy()

        end_x = frame.shape[1] - self.buffer
        end_y = self.buffer + self.draw_rectangle_height
        start_x = end_x - self.draw_rectangle_width
        start_y = end_y - self.draw_rectangle_height

        return end_x, end_y, start_x, start_y

    def draw_background_rectangle(self, frame):
        """
        Draws the background rectangle on the frame.

        :param frame: numpy.ndarray
            The input frame on which to draw the rectangle.

        :return: numpy.ndarray
            The frame with the background rectangle drawn on it.
        """
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_court(self, frame):
        """
        Draws the court on the given frame.

        :param frame: numpy.ndarray
            The input frame on which to draw the court.

        :return: numpy.ndarray
            The frame with the court drawn on it.
        """
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_mini_court(self, frames):
        """
        Draws the mini court on each frame in the given list of frames.

        :param frames: list of numpy.ndarray
            The list of frames on which to draw the mini court.

        :return: list of numpy.ndarray
            The list of frames with the mini court drawn on them.
        """
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)

        return output_frames

    def get_start_point_of_mini_court(self):
        """
        Gets the start point of the mini court.

        :return: tuple
            The x and y coordinates of the start point of the mini court.
        """
        return self.court_start_x, self.court_start_y

    def get_width_of_mini_court(self):
        """
        Gets the width of the mini court.

        :return: int
            The width of the mini court.
        """
        return self.court_drawing_width

    def get_court_drawing_key_points(self):
        """
        Gets the key points for drawing the court.

        :return: list of int
            A list of key points' coordinates for drawing the court.
        """
        return self.drawing_key_points

    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index, player_height_in_pixels, player_height_in_meters):
        """
        Converts the coordinates of an object to mini court coordinates.

        :param object_position: tuple
            The position of the object in the original court.
        :param closest_key_point: tuple
            The closest key point to the object in the original court.
        :param closest_key_point_index: int
            The index of the closest key point.
        :param player_height_in_pixels: int
            The height of the player in pixels.
        :param player_height_in_meters: float
            The height of the player in meters.

        :return: tuple
            The coordinates of the object in the mini court.
        """
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels, player_height_in_meters, player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels, player_height_in_meters, player_height_in_pixels)

        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = (self.drawing_key_points[closest_key_point_index * 2], self.drawing_key_points[closest_key_point_index * 2 + 1])

        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels, closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)

        return mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        """
        Converts the bounding boxes of players and the ball to mini court coordinates.

        :param player_boxes: list of dict
            A list of dictionaries containing the bounding boxes of players.
        :param ball_boxes: list of dict
            A list of dictionaries containing the bounding boxes of the ball.
        :param original_court_key_points: list of int
            The key points of the original court.

        :return: tuple
            A tuple containing the converted bounding boxes of players and the ball.
        """
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_key_point = (original_court_key_points[closest_key_point_index * 2], original_court_key_points[closest_key_point_index * 2 + 1])

                # Get player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position, closest_key_point, closest_key_point_index, max_player_height_in_pixels, player_heights[player_id])

                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get the closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    closest_key_point = (original_court_key_points[closest_key_point_index * 2], original_court_key_points[closest_key_point_index * 2 + 1])

                    mini_court_player_position = self.get_mini_court_coordinates(ball_position, closest_key_point, closest_key_point_index, max_player_height_in_pixels, player_heights[player_id])
                    output_ball_boxes.append({1: mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        """
        Draws points on the mini court for each frame in the given list of frames.

        :param frames: list of numpy.ndarray
            The list of frames on which to draw the points.
        :param positions: list of dict
            The positions of the points to be drawn on the mini court.
        :param color: tuple
            The color of the points to be drawn. Default is green.

        :return: list of numpy.ndarray
            The list of frames with the points drawn on them.
        """
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)
        return frames
