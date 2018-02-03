import cv2
import numpy as np
from features.feature_extractor import FeatureExtractor


class HistogramFeatureExtractor(FeatureExtractor):
    def __init__(self, num_bins=32, color_space="hsv"):
        self.num_bins = num_bins
        self.color_space = color_space

    def extract(self, image):
        transformed = image
        if self.color_space == "hsv":
            transformed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "hls":
            transformed = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif self.color_space == "luv":
            transformed = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif self.color_space == "yuv":
            transformed = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        hist_ch1 = np.histogram(transformed[:, :, 0], bins=self.num_bins, range=(0, 256))
        hist_ch2 = np.histogram(transformed[:, :, 1], bins=self.num_bins, range=(0, 256))
        hist_ch3 = np.histogram(transformed[:, :, 2], bins=self.num_bins, range=(0, 256))

        return np.concatenate((hist_ch1[0], hist_ch2[0], hist_ch3[0]))
