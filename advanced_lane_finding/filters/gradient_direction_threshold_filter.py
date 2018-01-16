import cv2
import numpy as np
from filters.image_filter import ImageFilter


class GradientDirectionThresholdFilter(ImageFilter):
    """
    Gradient threshold based on gradient direction.
    """
    def __init__(self, low_threshold, high_threshold, kernel_size=5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size

    def apply(self, image, context):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        grad_dir = np.arctan2(np.absolute(grad_y), np.absolute(grad_x))
        mask = (grad_dir >= self.low_threshold) \
            & (grad_dir <= self.high_threshold)
        binary_output = np.zeros_like(grad_dir, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        return '{}(lo:{}, hi:{}, ksize:{})'.format(
            self.__class__.__name__,
            self.low_threshold,
            self.high_threshold,
            self.kernel_size)
