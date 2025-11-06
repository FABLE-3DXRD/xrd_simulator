'''Script to create detector components to be used within the xrd_simulator package. 
They are stored by default in the /artifacts/detector/ folder within the current directory'''

import os
import numpy as np
from xrd_simulator.detector import Detector

# Detector parameters. Default is a Pilatus6M100k
samp_to_det_dist = 227_000 # um
pixel_size_z = 172. # um
pixel_size_y = 172. # um
det_width = 431_000 # um
det_height = 448_000 # um
destination = os.path.join('artifacts','detectors')
dir_this_file=os.path.dirname(os.path.abspath(__file__))
psf = None

detector = Detector(pixel_size_z=pixel_size_z,
                    pixel_size_y=pixel_size_y,
                    det_corner_0=np.array([samp_to_det_dist, -det_width*0.5, -det_height*0.5]),
                    det_corner_1=np.array([samp_to_det_dist, det_width*0.5, -det_height*0.5]),
                    det_corner_2=np.array([samp_to_det_dist, -det_width*0.5, det_height*0.5]))


if not os.path.exists(destination):
    os.makedirs(destination)

detector.save(os.path.join(dir_this_file,destination,f'{int(samp_to_det_dist*0.001)}mm_distance_{int(det_width*0.001)}mm_{int(det_height*0.001)}mm_size'))
print('Detector file created in ' + os.path.join(dir_this_file,destination,f'{int(samp_to_det_dist*0.001)}mm_distance_{int(det_width*0.001)}mm_{int(det_height*0.001)}mm_size.det'))