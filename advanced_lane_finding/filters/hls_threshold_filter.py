import cv2
import numpy as np
from filters.image_filter import ImageFilter


class HLSThresholdFilter(ImageFilter):
    """
    HLS based color space threshold.
    """
    def __init__(self, h_thresholds=(0, 255), l_thresholds=(0, 255), s_thresholds=(0, 255)):
        self.h_thresholds = h_thresholds
        self.l_thresholds = l_thresholds
        self.s_thresholds = s_thresholds

    def apply(self, image, context):
        rows, cols, ch = image.shape
        if ch != 3:
            raise ValueError("Expects 3 channel input")
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        h_ch = hls[:, :, 0]
        l_ch = hls[:, :, 1]
        s_ch = hls[:, :, 2]
        mask = (h_ch >= self.h_thresholds[0]) \
            & (h_ch <= self.h_thresholds[1]) \
            & (l_ch >= self.l_thresholds[0]) \
            & (l_ch <= self.l_thresholds[1]) \
            & (s_ch >= self.s_thresholds[0]) \
            & (s_ch <= self.s_thresholds[1])
        binary_output = np.zeros_like(s_ch, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        return '{}(h:{}, l:{}, s:{})'.format(
            self.__class__.__name__,
            self.h_thresholds,
            self.l_thresholds,
            self.s_thresholds)
