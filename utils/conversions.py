def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    """
    Converts a distance measured in pixels to meters using a reference object's height.

    :param pixel_distance: float
        The distance measured in pixels.

    :param reference_height_in_meters: float
        The known height of the reference object in meters.

    :param reference_height_in_pixels: float
        The height of the reference object in pixels.

    :return: float
        The distance converted to meters.
    """
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels


def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    """
    Converts a distance measured in meters to pixels using a reference object's height.

    :param meters: float
        The distance measured in meters.

    :param reference_height_in_meters: float
        The known height of the reference object in meters.

    :param reference_height_in_pixels: float
        The height of the reference object in pixels.

    :return: float
        The distance converted to pixels.
    """
    return (meters * reference_height_in_pixels) / reference_height_in_meters
