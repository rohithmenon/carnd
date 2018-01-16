import cv2
import numpy as np
from constants.constants import METERS_PER_PIXEL_Y, METERS_PER_PIXEL_X
from filters.image_filter import ImageFilter


class CurvatureDetailsFilter(ImageFilter):
    """
    Add Curvature details to the image.
    """
    def apply(self, image, context):
        if context:
            smoothed_left_lane_fit = context.smoothed_left_lane_fit()
            smoothed_right_lane_fit = context.smoothed_right_lane_fit()

            y_vals = np.linspace(0, 719, num=720)
            y_eval = np.max(y_vals) * METERS_PER_PIXEL_Y

            def evaluate(fit, value):
                return fit[0] * value ** 2 + fit[1] * value + fit[2]

            left_x = evaluate(smoothed_left_lane_fit, y_vals)
            right_x = evaluate(smoothed_right_lane_fit, y_vals)

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(y_vals * METERS_PER_PIXEL_Y, left_x * METERS_PER_PIXEL_X, 2)
            right_fit_cr = np.polyfit(y_vals * METERS_PER_PIXEL_Y, right_x * METERS_PER_PIXEL_X, 2)

            # Calculate the new radii of curvature
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

            # Now our radius of curvature is in meters and how far its off of center
            left_lane_x = evaluate(left_fit_cr, y_eval)
            right_lane_x = evaluate(right_fit_cr, y_eval)
            image_center = image.shape[1] * METERS_PER_PIXEL_X / 2
            lane_center = (left_lane_x + right_lane_x) / 2
            cv2.putText(image, "Left Curvature: {}".format(round(left_curverad, 2)), (800, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=3)
            cv2.putText(image, "Right Curvature: {}".format(round(right_curverad, 2)), (800, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=3)
            cv2.putText(image, "Center Deviation: {}".format(round(image_center - lane_center, 2)), (800, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=5)

        return image
