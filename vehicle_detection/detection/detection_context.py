import numpy as np

NUM_HEAT_MAP_FRAMES = 15
WINDOW_SIZE = 7


class DetectionContext(object):
    def __init__(self):
        self.heat_maps = np.array([])

    def add_heat_map(self, heat_map):
        new_heat_map = np.array([heat_map])
        self.heat_maps = np.append(self.heat_maps, new_heat_map, axis=0) if len(self.heat_maps) else new_heat_map
        self.heat_maps = self.heat_maps[-NUM_HEAT_MAP_FRAMES:, :, :]

    def smoothed_heat_map(self):
        # Smoothing is performed using the following logic:
        # 1. Compute average of the last WINDOW_SIZE heat_maps
        # 2. Slide the window until max windows (NUM_HEAT_MAP_FRAMES)
        # 3. Return the averaged heat_map that has the higest pixel value average.
        windows = np.array([np.mean(self.heat_maps[i:i + WINDOW_SIZE, :, :], 0)
                            for i in range(NUM_HEAT_MAP_FRAMES - WINDOW_SIZE)])
        means = np.array([np.mean(window) for window in windows])
        return windows[np.argmax(means)]
