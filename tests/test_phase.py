
import unittest
import numpy as np
from xrd_simulator.phase import Phase

class TestPhase(unittest.TestCase):

    def test_init(self):
        unit_cell = [5.028, 5.028, 5.519, 90., 90., 120.]
        sgname = "P3221"
        ph  = Phase(unit_cell, sgname)

    def test_generate_miller_indices(self):
            
        unit_cell = [5.028, 5.028, 5.519, 90., 90., 120.]
        sgname = "P3221"
        ph  = Phase(unit_cell, sgname)

        wavelength = 1.0
        min_bragg_angle = 1  * np.pi/180
        max_bragg_angle = 25 * np.pi/180

        hkl = ph.generate_miller_indices( wavelength, min_bragg_angle, max_bragg_angle )

        self.assertEqual(  hkl.shape[1], 3 )
        self.assertTrue(   hkl.shape[0] > 10 )

if __name__ == '__main__':
    unittest.main()
