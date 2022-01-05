import numpy as np
from xrd_simulator.detector import Detector

pixel_size_z = 75.0
pixel_size_y = 55.0
detector_size_z = pixel_size_z * 1024
detector_distance = 142938.3
d0 = np.array([detector_distance, -detector_size_z / 2., -detector_size_z / 2.])
d1 = np.array([detector_distance, detector_size_z / 2., -detector_size_z / 2.])
d2 = np.array([detector_distance, -detector_size_z / 2., detector_size_z / 2.])
detector = Detector(pixel_size_z, pixel_size_y, d0, d1, d2)
