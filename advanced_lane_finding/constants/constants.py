import numpy as np


CHESSBOARD_CORNERS = (9, 6)
IMAGE_SIZE_ROWS = 720
IMAGE_SIZE_COLS = 1280
IMAGE_SIZE = (IMAGE_SIZE_ROWS, IMAGE_SIZE_COLS)
BOTTOM_LEFT = (120, IMAGE_SIZE_ROWS)
BOTTOM_RIGHT = (1210, IMAGE_SIZE_ROWS)
TOP_RIGHT = (900, 550)
TOP_LEFT = (400, 550)

HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 10
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 150
MIN_LANE_SLOPE = 1.73
MAX_LANE_SLOPE = 60.0

LANE_REGION_BOTTOM_LEFT = (150, IMAGE_SIZE_ROWS)
LANE_REGION_BOTTOM_RIGHT = (1190, IMAGE_SIZE_ROWS)
LANE_REGION_TOP_RIGHT = (760, 450)
LANE_REGION_TOP_LEFT = (550, 450)

METERS_PER_PIXEL_Y = 30/720 # meters per pixel in y dimension
METERS_PER_PIXEL_X = 3.7/700 # meters per pixel in x dimension

def lane_region_vertices(image_shape):
    """
    Return the region defining lanes.
    """
    new_rows, new_cols = image_shape
    top_left_x, top_left_y = LANE_REGION_TOP_LEFT
    top_right_x, top_right_y = LANE_REGION_TOP_RIGHT
    bottom_left_x, bottom_left_y = LANE_REGION_BOTTOM_LEFT
    bottom_right_x, bottom_right_y = LANE_REGION_BOTTOM_RIGHT

    rows, cols = IMAGE_SIZE
    x_scale = new_cols / cols
    y_scale = new_rows / rows
    return [
        (int(bottom_left_x * x_scale), int(bottom_left_y * y_scale)),
        (int(bottom_right_x * x_scale), int(bottom_right_y * y_scale)),
        (int(top_right_x * x_scale), int(top_right_y * y_scale)),
        (int(top_left_x * x_scale), int(top_left_y * y_scale))]

def perspective_vertices(image_shape):
    """
    Return source region for  perspective
    """
    new_rows, new_cols = image_shape
    top_left_x, top_left_y = TOP_LEFT
    top_right_x, top_right_y = TOP_RIGHT
    bottom_left_x, bottom_left_y = BOTTOM_LEFT
    bottom_right_x, bottom_right_y = BOTTOM_RIGHT

    rows, cols = IMAGE_SIZE
    x_scale = new_cols / cols
    y_scale = new_rows / rows
    return [
        (int(bottom_left_x * x_scale), int(bottom_left_y * y_scale)),
        (int(bottom_right_x * x_scale), int(bottom_right_y * y_scale)),
        (int(top_right_x * x_scale), int(top_right_y * y_scale)),
        (int(top_left_x * x_scale), int(top_left_y * y_scale))]


def perspective_transformed_vertices(image_shape):
    """
    Return target region for  perspective
    """
    new_rows, new_cols = image_shape
    top_left_x, top_left_y = TOP_LEFT
    top_right_x, top_right_y = TOP_RIGHT
    bottom_left_x, bottom_left_y = BOTTOM_LEFT
    bottom_right_x, bottom_right_y = BOTTOM_RIGHT

    rows, cols = IMAGE_SIZE
    x_scale = new_cols / cols
    y_scale = new_rows / rows
    return [
        (int((bottom_left_x + 150) * x_scale), int(bottom_left_y * y_scale)),
        (int((bottom_right_x - 150) * x_scale), int(bottom_right_y * y_scale)),
        (int((bottom_right_x - 150) * x_scale), int(600 * y_scale)),
        (int((bottom_left_x  + 150) * x_scale), int(600 * y_scale))]
