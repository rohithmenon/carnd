import cv2
import numpy as np
from filters.image_filter import ImageFilter


class UndistortFilter(ImageFilter):
    """
    Image filter that performs camera undistorion.
    """
    def __init__(self, chessbord_images, corners_width=9, corners_height=6):
        width, height, ch = chessbord_images[0].shape
        obj_point = np.zeros((corners_width * corners_height, 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:corners_width, 0:corners_height].T.reshape(-1, 2)
        obj_points = []
        img_points = []
        for chessboard_image in chessbord_images:
            gray = cv2.cvtColor(chessboard_image, cv2.COLOR_RGB2GRAY) if ch == 3 else chessboard_image
            ret, corners = cv2.findChessboardCorners(gray, (corners_width, corners_height), None)
            if ret:
                obj_points.append(obj_point)
                img_points.append(corners)
        if not img_points:
            raise ValueError("Calibration images not present")
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)

    def apply(self, img, context):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def __repr__(self):
        return '{}(mtx:{}, dist:{})'.format(self.__class__.__name__, self.mtx, self.dist)
