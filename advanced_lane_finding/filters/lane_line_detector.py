import cv2
import numpy as np
from filters.filter_context import FrameData
from filters.image_filter import ImageFilter


class LaneLineDetector(ImageFilter):
    """
    Lane detector fiters
    """
    def find_lane_points(self, image, window_width, window_height, margin):
        """
        Function that searches for lane pixels without any previous knowledge of a lane line fit.
        """
        left_lane_points = []
        right_lane_points = []
        window = np.ones(window_width)
        rows, cols = image.shape

        l_sum = np.sum(image[int(2 * rows / 3):, :int(cols / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(2 * rows / 3):, int(cols / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(cols / 2)

        left_lane_points.append((l_center, rows - 1))
        right_lane_points.append((r_center, rows - 1))

        # Go through each layer looking for max pixel locations
        for layer in range(0, int(rows - window_height), 25):
            # convolve the window inxsto the vertical slice of the image
            layer_row_start = int(rows - (layer + window_height))
            layer_row_end = int(rows - layer)
            image_layer = np.sum(image[layer_row_start:layer_row_end, :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, cols))
            l_max_val = np.max(conv_signal[l_min_index:l_max_index])
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, cols))
            r_max_val = np.max(conv_signal[r_min_index:r_max_index])
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            if l_max_val > 0.0:
                left_lane_points.append((l_center, layer_row_start))
            if r_max_val > 0.0:
                right_lane_points.append((r_center, layer_row_start))

        return (
            self.fit_points(left_lane_points) if len(left_lane_points) > 3 else None,
            self.fit_points(right_lane_points) if len(right_lane_points) > 3 else None)

    def guided_find_lanes(self, image, prev_left_fit, prev_right_fit):
        """
        Function that searches for lane pixels with previous knowledge of a lane line fit.
        """
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        def evaluate(fit, value):
            return fit[0] * value ** 2 + fit[1] * value + fit[2]

        left_fitted = evaluate(prev_left_fit, nonzeroy)
        right_fitted = evaluate(prev_right_fit, nonzeroy)
        left_lane_inds = ((nonzerox > (left_fitted - margin)) & (nonzerox < left_fitted + margin))
        right_lane_inds = ((nonzerox > (right_fitted - margin)) & (nonzerox < right_fitted + margin))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 3 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 3 else None

        return (left_fit, right_fit)

    def fit_points(self, points):
        x_vals = []
        y_vals = []
        for point in points:
            x_val, y_val = point
            x_vals.append(x_val)
            y_vals.append(y_val)
        return np.polyfit(y_vals, x_vals, 2)

    def fitted_segments(self, fit, y_start, y_end, num_segments=50):
        c2, c1, c0 = fit
        new_y_vals = np.arange(y_start, y_end, (y_end - y_start) / num_segments)
        new_x_vals = [c2 * y**2 + c1 * y + c0 for y in new_y_vals]
        interpolated_line = []
        for i in range(1, len(new_x_vals), 2):
            interpolated_line.append(
                [new_x_vals[i - 1], new_y_vals[i - 1], new_x_vals[i], new_y_vals[i]])

        return interpolated_line

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(
                    img,
                    (int(x1 + 0.5), int(y1 + 0.5)),
                    (int(x2 + 0.5), int(y2 + 0.5)),
                    color,
                    thickness)

    def get_lane_vertices(self, l_lane_lines, r_lane_lines):
        vertices = []
        for line in l_lane_lines:
            x1, y1, x2, y2 = line
            vertices.append((int(x1 + 0.5), int(y1 + 0.5)))
            vertices.append((int(x2 + 0.5), int(y2 + 0.5)))

        for line in reversed(r_lane_lines):
            x1, y1, x2, y2 = line
            vertices.append((int(x2 + 0.5), int(y2 + 0.5)))
            vertices.append((int(x1 + 0.5), int(y1 + 0.5)))

        return vertices

    def apply(self, image, context):
        rows, cols = image.shape

        if context and context.smoothed_left_lane_fit() is not None and context.smoothed_right_lane_fit() is not None:
            (left_fit, right_fit) = self.guided_find_lanes(
                image,
                context.smoothed_left_lane_fit(),
                context.smoothed_right_lane_fit())
        else:
            (left_fit, right_fit) = self.find_lane_points(
                image, int(cols / 16), int(rows / 3), 25)

        if context:
            frame_data = FrameData()
            frame_data.set_left_lane_fit(left_fit)
            frame_data.set_right_lane_fit(right_fit)
            context.add_frame_data(frame_data)

        y_start = rows / 4
        y_end = rows

        lane_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        if left_fit is not None:
            l_interp = self.fitted_segments(left_fit, y_start, y_end)
            self.draw_lines(lane_img, np.array([l_interp]), thickness=50, color=255)
        if right_fit is not None:
            r_interp = self.fitted_segments(right_fit, y_start, y_end)
            self.draw_lines(lane_img, np.array([r_interp]), thickness=50, color=255)

        if left_fit is not None and right_fit is not None:
            lane_vertices = self.get_lane_vertices(l_interp, r_interp)
            cv2.fillPoly(
                lane_img,
                np.array([lane_vertices], dtype=np.dtype('int')),
                (128, 244, 65))
        return lane_img
