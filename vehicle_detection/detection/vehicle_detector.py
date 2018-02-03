import collections
import cv2
import numpy as np
from detection.image_window_vehicle_detector import ImageWindowVehicleDetector
from preprocess.preprocessor import PRE_PROCESS_IMAGE_SIZE
from scipy.ndimage.measurements import label
from threading import Thread
from queue import Queue

MIN_PIXEL_THRESHOLD = 4096
DetectionData = collections.namedtuple('DetectionData', 'detector window heat_map scale_factor')


class VehicleDetector(object):
    def __init__(
            self,
            model,
            scaler,
            window_sizes=(64, 96, 128, 160, 192),
            window_overlap=0.8,
            search_region=(0.55, 0.0, 1.0, 1.0),
            min_votes=2):
        self.model = model
        self.scaler = scaler
        self.window_sizes = window_sizes
        self.window_overlap = window_overlap
        self.search_region = search_region
        self.min_votes = min_votes
        self.worker_queue = Queue()
        for i in range(16):
            t = Thread(target=self.do_detection)
            t.daemon = True
            t.start()

    def do_detection(self):
        while True:
            detection_data = self.worker_queue.get()
            detector = detection_data.detector
            scaled_window = detection_data.window
            heat_map = detection_data.heat_map
            scale_factor = detection_data.scale_factor
            try:
                if detector.detect(scaled_window):
                    window = (np.array(scaled_window) / scale_factor).astype(int)
                    heat_map[window[0]:window[2], window[1]:window[3]] += 1
            finally:
                self.worker_queue.task_done()

    def add_detection(self, image, context=None):
        height, width, ch = image.shape
        heat_map = np.zeros((height, width))
        # For each window where a vehicle is detected increase window vote in heat map
        for window_size in self.window_sizes:
            scale_factor = PRE_PROCESS_IMAGE_SIZE / window_size
            scaled_image = cv2.resize(
                image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            scaled_windows = self.get_windows(scaled_image, PRE_PROCESS_IMAGE_SIZE)
            image_window_vehicle_detector = ImageWindowVehicleDetector(
                scaled_image, self.model, self.scaler)
            for scaled_window in scaled_windows:
                self.worker_queue.put(
                    DetectionData(image_window_vehicle_detector, scaled_window, heat_map, scale_factor))
        self.worker_queue.join()
        if context:
            context.add_heat_map(heat_map)
        smoothed_heat_map = context.smoothed_heat_map() if context else heat_map
        # Threshold heatmap by min window votes
        smoothed_heat_map[(smoothed_heat_map < self.min_votes)] = 0
        # Identify different vehicles from heat map
        vehicle_labels = label(smoothed_heat_map)
        vehicle_detected_img = np.copy(image)
        # For each detected vehicle draw a bounding box
        for vehicle_label in range(1, vehicle_labels[1] + 1):
            nonzero = (vehicle_labels[0] == vehicle_label).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = np.array(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
            if np.product(bbox[1] - bbox[0]) > MIN_PIXEL_THRESHOLD:
                # Draw the box on the image
                cv2.rectangle(vehicle_detected_img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 6)

        return vehicle_detected_img

    def get_windows(self, image, window_size):
        height, width, ch = image.shape
        search_start_height = int(height * self.search_region[0])
        search_start_width = int(width * self.search_region[1])
        search_end_height = int(height * self.search_region[2])
        search_end_width = int(width * self.search_region[3])

        windows = []
        pixel_step = int(window_size * (1.0 - self.window_overlap))
        for w in range(search_start_width, search_end_width - window_size, pixel_step):
            windows.append((
                search_start_height,
                w,
                min(search_start_height + window_size, search_end_height),
                w + window_size))
        return windows
