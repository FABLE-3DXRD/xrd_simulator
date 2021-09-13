import unittest
import numpy as np
from xrd_simulator.scatterer import Scatterer
from scipy.spatial import ConvexHull

class TestScatterer(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)
        wavelength = 1.0
        verts = np.array([[0,0,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0],
                          [0.1 ,0.1, 0.1]])
        self.ch = ConvexHull( verts )
        kprime = np.random.rand(3,)
        self.kprime = 2*np.pi*kprime/(wavelength * np.linalg.norm(kprime) )
        self.s = np.random.rand()
        self.scatterer = Scatterer(self.ch, self.kprime, self.s)

    def test_get_centroid(self):
        centroid2 = self.scatterer.get_centroid()
        for c1,c2 in zip(np.array([0.25, 0.25, 0.25]), centroid2):
            self.assertAlmostEqual( c1, c2, msg="Centroid is wrong" )

    def test_get_volume(self):
        vol = self.scatterer.get_volume()
        self.assertAlmostEqual( vol, 1/6., msg="Centroid is wrong" )

if __name__ == '__main__':
    unittest.main()