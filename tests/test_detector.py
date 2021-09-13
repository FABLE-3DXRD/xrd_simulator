import unittest
import numpy as np
from xrd_simulator.detector import Detector
from xrd_simulator.scatterer import Scatterer
from scipy.spatial import ConvexHull

class TestDetector(unittest.TestCase):

    def setUp(self):
        self.pixel_size = 50.
        self.detector_size = 100000.
        geometry_matrix_0 = np.array([[1,0,0],[1,1,0],[1,0,1]]).T*self.detector_size
        def geometry_matrix(s): 
            sin = np.sin( -s*np.pi/2. )
            cos = np.cos( -s*np.pi/2. )
            Rz = np.array([ [ cos, -sin, 0 ],
                            [ sin,  cos, 0 ],
                            [  0,    0,  1 ] ])
            return Rz.dot( geometry_matrix_0 )
        self.detector = Detector( self.pixel_size, geometry_matrix )

    def test_init(self):
        for i in range(3):
            n = self.detector.normalised_geometry_matrix[:,i]
            self.assertAlmostEqual(np.linalg.norm(n), 1 ,msg="normalised_geometry_matrix is not normalised")

        for z,ztrue in zip(self.detector.zdhat, np.array([0,0,1])):
            self.assertAlmostEqual(z, ztrue ,msg="zdhat is incorrect")

        for y,ytrue in zip(self.detector.ydhat, np.array([0,1,0])):
            self.assertAlmostEqual(y, ytrue ,msg="ydhat is incorrect")

        self.assertAlmostEqual(self.detector.zmax, self.detector_size, msg="Bad detector dimensions in zmax")
        self.assertAlmostEqual(self.detector.ymax, self.detector_size, msg="Bad detector dimensions in ymax")

        for n,ntrue in zip(self.detector.normal, np.array([-1,0,0])):
            self.assertAlmostEqual(n, ntrue ,msg="Bad detector normal")

    def test_set_geometry(self):
        self.detector.set_geometry(s=1)

        for i in range(3):
            n = self.detector.normalised_geometry_matrix[:,i]
            self.assertAlmostEqual(np.linalg.norm(n), 1 ,msg="normalised_geometry_matrix is not normalised")

        for z,ztrue in zip(self.detector.zdhat, np.array([0,0,1])):
            self.assertAlmostEqual(z, ztrue ,msg="zdhat is incorrect")

        for y,ytrue in zip(self.detector.ydhat, np.array([1,0,0])):
            self.assertAlmostEqual(y, ytrue ,msg="ydhat is incorrect")

        self.assertAlmostEqual(self.detector.zmax, self.detector_size, msg="Bad detector dimensions in zmax")
        self.assertAlmostEqual(self.detector.ymax, self.detector_size, msg="Bad detector dimensions in ymax")

        for n,ntrue in zip(self.detector.normal, np.array([0,1,0])):
            self.assertAlmostEqual(n, ntrue ,msg="Bad detector normal")

    def test_render(self):
        np.random.seed(10)
        wavelength = 1.0
        verts = np.array([[0,0,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0],
                          [0.1 ,0.1, 0.1]])*5*self.pixel_size
        ch = ConvexHull( verts )
        kprime = 2*np.pi*np.array([1,0,0])/(wavelength)
        scatterer = Scatterer(ch, kprime, s=0)
        self.detector.frames.append([scatterer])
        piximage = self.detector.render(frame_number=0)
        self.assertAlmostEqual(piximage[1000,1000], ch.volume)

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