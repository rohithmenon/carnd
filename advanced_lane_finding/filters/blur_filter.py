import cv2
from filters.image_filter import ImageFilter


class BlurFilter(ImageFilter):
    """
    Image filter that blurs using bilateral filter.
    """
    def __init__(self, diameter=9, sigma=25):
        self.diameter = diameter
        self.sigma = sigma

    def apply(self, image, context):
        return cv2.bilateralFilter(image, self.diameter, self.sigma, self.sigma)
