import cv2
import numpy as np
from constants.constants import lane_region_vertices
from filters.image_filter import ImageFilter


class LaneRegionMaskFilter(ImageFilter):
    """
    Filter that applies a lane region mask.
    """
    def apply(self, image, context):
        mask = np.zeros_like(image, np.uint8)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(
            mask,
            np.array([lane_region_vertices(image.shape)], dtype=np.dtype('int')),
            ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        #masked_image = cv2.addWeighted(image, 1.0, mask, 0.005, 0.0)

        return masked_image
