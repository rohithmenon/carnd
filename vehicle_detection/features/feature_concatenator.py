import numpy as np
from features.feature_extractor import FeatureExtractor


class FeatureConcatenator(FeatureExtractor):
    def __init__(self, extractors):
        self.extractors = extractors

    def extract_window(self, image, window):
        features = []
        for extractor in self.extractors:
            features.append(extractor.extract_window(image, window))
        return np.concatenate(features)

    def extract(self, image):
        features = []
        for extractor in self.extractors:
            features.append(extractor.extract_window(image))
        return np.concatenate(features)
