def get_center_of_box(box):
    """
    Calculates the center point of a bounding box.

    :param box: list or tuple
        A list or tuple containing four coordinates (x1, y1, x2, y2).

    :return: tuple
        The (x, y) coordinates of the center point of the bounding box.
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def measure_distance(p1,p2):
    """
    Measures the Euclidean distance between two points.

    :param p1: tuple
        The (x, y) coordinates of the first point.

    :param p2: tuple
        The (x, y) coordinates of the second point.

    :return: float
        The Euclidean distance between the two points.
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def get_foot_position(bbox):
    """
    Gets the foot (bottom-center) position of a bounding box.

    :param bbox: list or tuple
        A list or tuple containing four coordinates (x1, y1, x2, y2).

    :return: tuple
        The (x, y) coordinates of the foot position of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, key_points, keypoint_indices):
    """
    Finds the index of the closest key point to a given point.

    :param point: tuple
        The (x, y) coordinates of the point.

    :param key_points: list
        A list of key points, each represented by two consecutive elements (x, y).

    :param keypoint_indices: list
        A list of indices specifying which key points to consider.

    :return: int
        The index of the closest key point.
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = key_points[keypoint_index * 2], key_points[keypoint_index * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index

    return key_point_ind


def get_height_of_bbox(bbox):
    """
    Calculates the height of a bounding box.

    :param bbox: list or tuple
        A list or tuple containing four coordinates (x1, y1, x2, y2).

    :return: int
        The height of the bounding box.
    """
    return bbox[3]-bbox[1]


def measure_xy_distance(p1, p2):
    """
    Measures the distance between two points along the x and y axes.

    :param p1: tuple
        The (x, y) coordinates of the first point.

    :param p2: tuple
        The (x, y) coordinates of the second point.

    :return: tuple
        The distances along the x-axis and y-axis between the two points.
    """
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_center_of_bbox(bbox):
    """
    Calculates the center point of a bounding box.

    :param bbox: list or tuple
        A list or tuple containing four coordinates (x1, y1, x2, y2).

    :return: tuple
        The (x, y) coordinates of the center point of the bounding box.
    """
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
