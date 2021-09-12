import unittest
import numpy as np
from xrd_simulator.detector import Detector

class TestDetector(unittest.TestCase):

    def setUp(self):
        pixel_size = 50.
        geometry_matrix_0 = np.array([[1,0,0],[1,1,0],[1,0,1]])*100000.
        def geometry_matrix(s): 
            sin = np.sin( s*np.pi/2. )
            cos = np.cos( s*np.pi/2. )
            Rz = np.array([ [ cos, -sin, 0 ],
                            [ sin,  cos, 0 ],
                            [  0,    0,  1 ] ])
            return Rz.dot( geometry_matrix_0 )
        detector = Detector( pixel_size, geometry_matrix )

    def test_init(self):
        pass

    def test_set_geometry(self):
        pass

    def test_render(self):
        pass

    def test_get_intersection(self):
        pass

    def test_contains(self):
        pass

    def test_get_wrapping_cone(self):
        pass

    def test_approximate_wrapping_cone(self):
        pass

if __name__ == '__main__':
    unittest.main()