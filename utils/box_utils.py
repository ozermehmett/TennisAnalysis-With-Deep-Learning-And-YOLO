def get_center_of_box(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, key_points, keypoint_indices):
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
    return bbox[3]-bbox[1]


def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])


def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
