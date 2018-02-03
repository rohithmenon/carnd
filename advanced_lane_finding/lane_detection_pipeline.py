import cv2
import glob
from constants.constants import CHESSBOARD_CORNERS
from filters.blur_filter import BlurFilter
from filters.grayscale_filter import GrayscaleFilter
from filters.gradient_direction_threshold_filter import GradientDirectionThresholdFilter
from filters.gradient_magnitude_threshold_filter import GradientMagnitudeThresholdFilter
from filters.gradient_threshold_filter import GradientThresholdFilter
from filters.hls_threshold_filter import HLSThresholdFilter
from filters.s_channel_filter import SaturationChannelFilter
from filters.lane_region_mask_filter import LaneRegionMaskFilter
from filters.lane_perspective_transform_filter import LanePerspectiveTransformFilter
from filters.undistort_filter import UndistortFilter
from filters.filter_ops import and_filters, weighted_and_filters
from filters.lane_line_detector import LaneLineDetector
from filters.curvature_details_filter import CurvatureDetailsFilter
from filters.filter_chain import FilterChain


class LaneDetector(object):
    def __init__(self, camera_calibration_image_path):
        # Undistort filter
        chessboard_imgs = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in glob.glob(camera_calibration_image_path)]
        undistort_filter = UndistortFilter(chessboard_imgs, CHESSBOARD_CORNERS[0], CHESSBOARD_CORNERS[1])
        # Blur filter
        blur_filter = BlurFilter()
        # Grayscale filter
        grayscale_filter = GrayscaleFilter(low_threshold=150, high_threshold=255)
        # SaturationChannelFilter
        s_channel_filter = SaturationChannelFilter()
        # HSL filter
        s_threshold_filter = HLSThresholdFilter(h_thresholds=(0, 255), l_thresholds=(0, 255), s_thresholds=(70, 255))
        # Gradient threshold filter
        gradx_threshold_filter = GradientThresholdFilter(low_threshold=5, high_threshold=255, orientation='x')
        # Gradient magnitude filter
        grad_mag_threshold_filter = GradientMagnitudeThresholdFilter(low_threshold=5, high_threshold=255)
        # Gradient direction filter
        grad_dir_threshold_filter = GradientDirectionThresholdFilter(low_threshold=0.7, high_threshold=1.2)
        # Lane Region mask filter
        lane_region_mask_filter = LaneRegionMaskFilter()
        lane_perpective_transform_filter = LanePerspectiveTransformFilter((720, 1280))
        inverse_lane_perpective_transform_filter = LanePerspectiveTransformFilter((720, 1280), inverse=True)
        lane_line_detector = LaneLineDetector()
        curvature_details_filter = CurvatureDetailsFilter()

        composite_gradient_filter = and_filters([
            gradx_threshold_filter,
            grad_mag_threshold_filter,
            grad_dir_threshold_filter
        ])

        self.chain = FilterChain([
            undistort_filter,
            blur_filter,
            weighted_and_filters([
                s_threshold_filter,
                FilterChain([
                    grayscale_filter,
                    composite_gradient_filter
                ]),
                FilterChain([
                    s_threshold_filter,
                    composite_gradient_filter
                ]),
                FilterChain([
                    s_channel_filter,
                    composite_gradient_filter
                ])
            ], 0.25),
            lane_region_mask_filter,
            lane_perpective_transform_filter,
            blur_filter,
            lane_line_detector,
            inverse_lane_perpective_transform_filter,
            curvature_details_filter
        ])

    # Function that detects and superimposes lanes on images.
    def detect_lanes(self, img, context=None):
        return self.chain.apply(img, context)
