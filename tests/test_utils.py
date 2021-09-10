import unittest
import numpy as np
from xrd_simulator import utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test

    def test_PlanarRodriguezRotator(self):

        wavelength = np.random.rand()*0.5
        k1 = (np.random.rand(3,)-0.5)*2
        k2 = (np.random.rand(3,)-0.5)*2
        k1 = 2*np.pi*k1/(np.linalg.norm(k1)*wavelength)
        k2 = 2*np.pi*k2/(np.linalg.norm(k2)*wavelength)

        rotator    = utils.PlanarRodriguezRotator( k1, k2 )
        k1dotk2 = np.cos(rotator.alpha)*4*np.pi*np.pi/(wavelength*wavelength)

        self.assertAlmostEqual( rotator.rhat.dot(k1), 0, msg="rhat is not orthogonal to k1" )
        self.assertAlmostEqual( rotator.rhat.dot(k2), 0, msg="rhat is not orthogonal to k2" )
        self.assertAlmostEqual( np.linalg.norm(rotator.rhat), 1.0, msg="rhat is not normalised" )
        self.assertLessEqual( rotator.alpha, np.pi, msg="alpha is greater than pi" )
        self.assertGreaterEqual( rotator.alpha, 0, msg="alpha is negative" )
        self.assertAlmostEqual( k1dotk2, k1.dot(k2), msg="alpha does not match dot product cosine formula" )

        k1rot = rotator(k1,s=0)
        for i in range(3):
            self.assertAlmostEqual( k1rot[i], k1[i], msg="Rotator does not map k1 to k1 for s=0" )

        k2rot = rotator(k1,s=1)
        for i in range(3):
            self.assertAlmostEqual( k2rot[i], k2[i], msg="Rotator does not map k2 to k2 for s=1" )

        krot = rotator(k1,s=0.5)
        halfalpha1 = np.arccos( krot.dot(k1)/(np.linalg.norm(k1)*np.linalg.norm(krot)) )
        self.assertAlmostEqual( rotator.alpha/2., halfalpha1, msg="Angle between k1 and a rotated vector is not alpha/2 fr s=0.5" )
        halfalpha2 = np.arccos( krot.dot(k2)/(np.linalg.norm(k2)*np.linalg.norm(krot)) )
        self.assertAlmostEqual( rotator.alpha/2., halfalpha2, msg="Angle between k2 and a rotated vector is not alpha/2 fr s=0.5" )

    def test_get_unit_vector_and_l2norm(self):
        point_1 = np.random.rand(3,)
        point_2 = np.random.rand(3,)
        unitvector, norm = utils.get_unit_vector_and_l2norm(point_1, point_2)
        self.assertAlmostEqual( np.linalg.norm(unitvector), 1, msg="unitvector is not unit length" )
        self.assertLessEqual( norm, np.sqrt(3), msg="norm is too large" )

if __name__ == '__main__':
    unittest.main()