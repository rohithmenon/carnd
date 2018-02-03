import numpy as np
from features.feature_concatenator import FeatureConcatenator
from features.feature_extractor import FeatureExtractor
from features.feature_normalizer import FeatureNormalizer
from features.hog_feature_extractor import HogFeatureExtractor
from features.histogram_feature_extractor import HistogramFeatureExtractor
from features.pooled_pixel_feature_extractor import PooledPixelFeatureExtractor


class ImageWindowHogFeatureExtractor(FeatureExtractor):
    """
    Calculate hog features once per image and extract hog features for window
    by reusing the image hog features.
    """
    def __init__(self, image):
        self.image = image
        hog_extractor = HogFeatureExtractor(feature_vector=False)
        self.hog_features = hog_extractor.extract(image)
        self.pixels_per_cell = hog_extractor.pixels_per_cell
        self.cells_per_block = hog_extractor.cells_per_block

    def extract_window(self, image, window):
        assert image is self.image
        hog_window = (np.array(window) / self.pixels_per_cell).astype(int)
        return (self.hog_features[hog_window[0]:hog_window[2] - 1, hog_window[1]:hog_window[3] - 1]).ravel()

    def extract(self, image):
        assert image is self.image
        return self.hog_features.ravel()


class ImageWindowVehicleDetector(object):
    """
    Detect car within a window over an image using the passed in classifier.
    """
    def __init__(self, image, model, scaler):
        self.image = image
        self.model = model
        histogram_extractor = HistogramFeatureExtractor()
        pooled_pixel_extractor = PooledPixelFeatureExtractor()
        hog_feature_extractor = ImageWindowHogFeatureExtractor(image)
        feature_concatenator = FeatureConcatenator(
            [histogram_extractor, hog_feature_extractor, pooled_pixel_extractor])
        self.feature_extractor = FeatureNormalizer(feature_concatenator, scaler)

    def detect(self, window):
        features = np.array(self.feature_extractor.extract_window(self.image, window)).reshape(1, -1)
        return self.model.predict(features) == 1
