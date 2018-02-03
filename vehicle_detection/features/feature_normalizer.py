import numpy as np
from features.feature_extractor import FeatureExtractor


class FeatureNormalizer(FeatureExtractor):
    def __init__(self, base_extractor, scaler):
        self.base_extractor = base_extractor
        self.scaler = scaler

    def extract_window(self, image, window):
        features = np.array(self.base_extractor.extract_window(image, window)).reshape(1, -1)
        return self.scaler.transform(features)[0]

    def extract(self, image, window):
        features = np.array(self.base_extractor.extract(image)).reshape(1, -1)
        return self.scaler.transform(features)[0]
