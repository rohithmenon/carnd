import cv2
import numpy as np
from filters.image_filter import ImageFilter


class GrayscaleFilter(ImageFilter):
    """
    Grayscale value based threshold filter.
    """
    def __init__(self, low_threshold=0, high_threshold=255):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, image, context):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else image
        ret_img = np.copy(gray)
        indices = (gray < self.low_threshold) & (gray > self.high_threshold)
        ret_img[indices] = 0
        return ret_img
