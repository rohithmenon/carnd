import cv2
import numpy as np
from constants.constants import perspective_vertices, perspective_transformed_vertices
from filters.image_filter import ImageFilter


class LanePerspectiveTransformFilter(ImageFilter):
    """
    Image filter that performs perspective transformation.
    """
    def __init__(self, image_size, inverse=False):
        src_points = np.float32(perspective_vertices(image_size))
        dst_points = np.float32(perspective_transformed_vertices(image_size))
        self.transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points) \
            if inverse else cv2.getPerspectiveTransform(src_points, dst_points)

    def apply(self, image, context):
        rows, cols = image.shape[0], image.shape[1]
        return cv2.warpPerspective(
            image,
            self.transform_matrix,
            (cols, rows),
            flags=cv2.INTER_LINEAR)
