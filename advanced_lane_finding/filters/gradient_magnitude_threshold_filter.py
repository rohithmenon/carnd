import cv2
import numpy as np
from filters.image_filter import ImageFilter


class GradientMagnitudeThresholdFilter(ImageFilter):
    """
    Gradient threshold based on gradient magnitude.
    """
    def __init__(self, low_threshold, high_threshold, kernel_size=5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size

    def apply(self, image, context):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        grad_xy = np.sqrt(grad_x ** 2 + grad_y ** 2)
        max_grad = np.max(grad_xy)
        scaled_grad = grad_xy if max_grad == 0 else np.uint8(255 * grad_xy / max_grad)
        mask = (scaled_grad >= self.low_threshold) \
            & (scaled_grad <= self.high_threshold)
        binary_output = np.zeros_like(scaled_grad, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        return '{}(lo:{}, hi:{}, ksize={})'.format(
            self.__class__.__name__,
            self.low_threshold,
            self.high_threshold,
            self.kernel_size)
