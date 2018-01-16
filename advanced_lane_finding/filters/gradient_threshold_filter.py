import cv2
import numpy as np
from filters.image_filter import ImageFilter


class GradientThresholdFilter(ImageFilter):
    """
    Gradient threshold based on gradient value.
    """
    def __init__(self, low_threshold, high_threshold, orientation='x', kernel_size=5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.orientation = orientation
        self.kernel_size = kernel_size

    def apply(self, image, context):
        grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.kernel_size) \
            if self.orientation == 'x' else cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        grad_abs = np.absolute(grad)
        max_grad = np.max(grad_abs)
        scaled_grad = grad_abs if max_grad == 0 else np.uint8(255 * grad_abs / max_grad)
        mask = (scaled_grad >= self.low_threshold) \
            & (scaled_grad <= self.high_threshold)
        binary_output = np.zeros(scaled_grad.shape, dtype=np.uint8)
        binary_output[mask] = 1
        return binary_output

    def __repr__(self):
        return '{}(lo:{}, hi:{}, orient:{}, ksize={})'.format(
            self.__class__.__name__,
            self.low_threshold,
            self.high_threshold,
            self.orientation,
            self.kernel_size)
