import unittest
import numpy as np
from xrd_simulator import utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test

    def test_get_unit_vector_and_l2norm(self):
        point_1 = np.random.rand(3,)
        point_2 = np.random.rand(3,)
        unitvector, norm = utils.get_unit_vector_and_l2norm(point_1, point_2)
        self.assertAlmostEqual( np.linalg.norm(unitvector), 1, msg="unitvector is not unit length" )
        self.assertLessEqual( norm, np.sqrt(3), msg="norm is too large" )

if __name__ == '__main__':
    unittest.main()