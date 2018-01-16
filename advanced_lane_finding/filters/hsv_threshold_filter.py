import cv2
import numpy as np
from filters.image_filter import ImageFilter


class HSVThresholdFilter(ImageFilter):
    """
    HSV based color space threshold.
    """
    def __init__(self, h_thresholds=(0, 255), s_thresholds=(0, 255), v_thresholds=(0, 255)):
        self.h_thresholds = h_thresholds
        self.s_thresholds = s_thresholds
        self.v_thresholds = v_thresholds

    def apply(self, image, context):
        rows, cols, ch = image.shape
        if ch != 3:
            raise ValueError("Expects 3 channel input")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        mask = (h_ch >= self.h_thresholds[0]) \
            & (h_ch <= self.h_thresholds[1]) \
            & (s_ch >= self.s_thresholds[0]) \
            & (s_ch <= self.s_thresholds[1]) \
            & (v_ch >= self.v_thresholds[0]) \
            & (v_ch <= self.v_thresholds[1])
        binary_output = np.zeros_like(s_ch, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        return '{}(h:{}, s:{}, v:{})'.format(
            self.__class__.__name__,
            self.h_thresholds,
            self.s_thresholds,
            self.v_thresholds)
