import unittest
import numpy as np
import os
from xrd_simulator import templates

class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test

    def test_s3dxrd(self):

        parameters = {
            "detector_distance"             : 191023.9164,
            "detector_center_pixel_z"       : 501.2345,
            "detector_center_pixel_y"       : 505.1129,
            "pixel_side_length_z"           : 181.4234,
            "pixel_side_length_y"           : 180.2343,
            "number_of_detector_pixels_z"   : 512,
            "number_of_detector_pixels_y"   : 512,
            "wavelength"                    : 0.285227,
            "beam_side_length_z"            : 400.3455,
            "beam_side_length_y"            : 500.4545,
            "rotation_step"                 : 1.0,
            "rotation_axis"                 : np.array([0., 0., 1.0])
        }

        beam, detector, motion = templates.s3dxrd( parameters )
        #sample = ...
        #sample.diffract( beam, detector, motion )
        #detector.render()

    def test_polycrystal_from_orientation_density(self):

        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221' # Quartz
        orientation_density_function = lambda q: q[0]/10.
        number_of_crystals = 50
        sample_bounding_cylinder_height = 100
        sample_bounding_cylinder_radius = 25

        polycrystal = templates.polycrystal_from_orientation_density(  orientation_density_function,
                                                                        number_of_crystals,
                                                                        sample_bounding_cylinder_height,
                                                                        sample_bounding_cylinder_radius,                                          
                                                                        unit_cell,
                                                                        sgname )

if __name__ == '__main__':
    unittest.main()