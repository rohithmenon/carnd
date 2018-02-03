class FeatureExtractor(object):
    def extract_window(self, image, window=None):
        """
        Extract features for a window over the passed image.
        """
        window_image = image[window[0]:window[2], window[1]:window[3], :] if window else image
        return self.extract(window_image)

    def extract(self, image):
        """
        Extract features for the passed image.
        """
        raise NotImplementedError()
