class FeatureExtractor(object):
    def extract_window(self, image, window=None):
        window_image = image[window[0]:window[2], window[1]:window[3], :] if window else image
        return self.extract(window_image)

    def extract(self, image):
        raise NotImplementedError()
