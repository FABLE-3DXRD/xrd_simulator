import numpy as np
from xrd_simulator.detector import Detector

pixel_size = 75.
detector_size = pixel_size * 1024
detector_distance = 142938.3
d0 = np.array([detector_distance, -detector_size / 2., -detector_size / 2.])
d1 = np.array([detector_distance,  detector_size / 2., -detector_size / 2.])
d2 = np.array([detector_distance, -detector_size / 2.,  detector_size / 2.])
detector = Detector(pixel_size, pixel_size, d0, d1, d2)