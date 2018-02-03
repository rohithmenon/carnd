import cv2
from features.feature_extractor import FeatureExtractor


class PooledPixelFeatureExtractor(FeatureExtractor):
    def __init__(self, sample_size=(32, 32)):
        self.sample_size = sample_size

    def extract(self, image):
        return cv2.resize(image, self.sample_size).ravel()
