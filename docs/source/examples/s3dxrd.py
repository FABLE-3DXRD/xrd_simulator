import numpy as np
from xrd_simulator import templates

parameters = {
    "detector_distance"             : 191023.9164,
    "detector_center_pixel_z"       : 256.2345,
    "detector_center_pixel_y"       : 255.1129,
    "pixel_side_length_z"           : 181.4234,
    "pixel_side_length_y"           : 180.2343,
    "number_of_detector_pixels_z"   : 512,
    "number_of_detector_pixels_y"   : 512,
    "wavelength"                    : 0.285227,
    "beam_side_length_z"            : 512 * 200.,
    "beam_side_length_y"            : 512 * 200.,
    "rotation_step"                 : 1.0,
    "rotation_axis"                 : np.array([0., 0., 1.0])
}

beam, detector, motion = templates.s3dxrd( parameters )