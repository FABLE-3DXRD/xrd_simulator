import numpy as np
from xrd_simulator.detector import Detector

# The detector plane is defined by it's corner coordinates det_corner_0,det_corner_1,det_corner_2
detector = Detector(pixel_size_z=75.0,
                    pixel_size_y=55.0,
                    det_corner_0=np.array([142938.3, -38400., -38400.]),
                    det_corner_1=np.array([142938.3, 38400., -38400.]),
                    det_corner_2=np.array([142938.3, -38400., 38400.]))
