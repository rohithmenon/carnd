import cv2
import glob
images = [cv2.imread(img_path) for img_path in glob.glob('camera_cal/*.jpg')]
from camera_calibrator import CameraCalibrator
calibrator = CameraCalibrator(images, 9, 6)
