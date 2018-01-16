import cv2
from filters.image_filter import ImageFilter


class CLAHEFilter(ImageFilter):
    """
    Image filter to perform Adaptive Contrast enhancement.
    """
    def __init__(self, clip_limit=2.0, tile_grid=(5, 5)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def apply(self, image, context):
        r_ch = image[:, :, 0]
        g_ch = image[:, :, 1]
        b_ch = image[:, :, 2]
        r_enhanced_ch = self.clahe.apply(r_ch)
        g_enhanced_ch = self.clahe.apply(g_ch)
        b_enhanced_ch = self.clahe.apply(b_ch)
        return cv2.merge((r_enhanced_ch, g_enhanced_ch, b_enhanced_ch))
