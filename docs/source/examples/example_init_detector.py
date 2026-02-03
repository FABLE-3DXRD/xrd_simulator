import numpy as np
import os
from xrd_simulator.detector import Detector

# The detector is defined by it's corner coordinates det_corner_0, det_corner_1, det_corner_2
# Option 1: Specify number of pixels directly
detector = Detector(det_corner_0=np.array([142938.3, -38400., -38400.]),
                    det_corner_1=np.array([142938.3, 38400., -38400.]),
                    det_corner_2=np.array([142938.3, -38400., 38400.]),
                    n_pixels=(1024, 1396))  # (n_z, n_y) pixels

# Option 2: Specify pixel size - single value for square pixels
detector = Detector(det_corner_0=np.array([142938.3, -38400., -38400.]),
                    det_corner_1=np.array([142938.3, 38400., -38400.]),
                    det_corner_2=np.array([142938.3, -38400., 38400.]),
                    pixel_size=75.0)  # 75 µm square pixels

# Option 3: Specify pixel size - tuple for rectangular pixels
detector = Detector(det_corner_0=np.array([142938.3, -38400., -38400.]),
                    det_corner_1=np.array([142938.3, 38400., -38400.]),
                    det_corner_2=np.array([142938.3, -38400., 38400.]),
                    pixel_size=(75.0, 55.0))  # (z, y) in µm

# The detector may be saved to disc for later usage.
artifacts_dir = os.path.join(os.path.dirname(__file__), 'test_artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
detector.save(os.path.join(artifacts_dir, 'my_detector'))
detector_loaded_from_disc = Detector.load(os.path.join(artifacts_dir, 'my_detector.det'))
