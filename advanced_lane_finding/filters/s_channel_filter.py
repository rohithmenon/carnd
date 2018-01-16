import cv2
from filters.image_filter import ImageFilter


class SaturationChannelFilter(ImageFilter):
    """
    Filter that extracts saturation channel from HLS color space.
    """
    def apply(self, image, context):
        if len(image.shape) < 3:
            raise ValueError("Expects 3 channel input")
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        return hls[:, :, 2]
