import cv2
from features.feature_extractor import FeatureExtractor
from skimage.feature import hog


class HogFeatureExtractor(FeatureExtractor):
    def __init__(self, pixels_per_cell=8, cells_per_block=2, feature_vector=True):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.feature_vector = feature_vector

    def extract(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
            cells_per_block=(self.cells_per_block, self.cells_per_block),
            visualise=False,
            feature_vector=False,
            block_norm="L2-Hys")
        return hog_features.ravel() if self.feature_vector else hog_features
