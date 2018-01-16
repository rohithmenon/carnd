import numpy as np


class FrameData(object):
    def __init__(self):
        self.left_lane_fit = None
        self.right_lane_fit = None

    def set_left_lane_fit(self, fit):
        self.left_lane_fit = fit

    def set_right_lane_fit(self, fit):
        self.right_lane_fit = fit

    def get_left_lane_fit(self):
        return self.left_lane_fit

    def get_right_lane_fit(self):
        return self.right_lane_fit


class FilterContext(object):
    """
    Context object to hold details about line fits across multiple
    video frame processing.
    """
    def __init__(self, debug_fn=None):
        self.frame_data_list = []
        self.debug_fn = debug_fn

    def add_frame_data(self, frame_data):
        self.frame_data_list.append(frame_data)
        self.frame_data_list = self.frame_data_list[-10:]

    def smoothed_left_lane_fit(self):
        num_points = 0
        sum_fit = np.array([0.0, 0.0, 0.0])
        for frame_data in self.frame_data_list:
            if frame_data and frame_data.get_left_lane_fit() is not None:
                sum_fit = sum_fit + frame_data.get_left_lane_fit()
                num_points += 1
        return sum_fit / num_points if num_points else None

    def smoothed_right_lane_fit(self):
        num_points = 0
        sum_fit = np.array([0.0, 0.0, 0.0])
        for frame_data in self.frame_data_list:
            if frame_data and frame_data.get_right_lane_fit() is not None:
                sum_fit = sum_fit + frame_data.get_right_lane_fit()
                num_points += 1
        return sum_fit / num_points if num_points else None
