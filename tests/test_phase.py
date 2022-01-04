
import os
import unittest
import numpy as np
from xrd_simulator.phase import Phase


class TestPhase(unittest.TestCase):

    def test_init(self):
        unit_cell = [5.028, 5.028, 5.519, 90., 90., 120.]
        sgname = "P3221"
        ph = Phase(unit_cell, sgname)
        self.assertTrue(ph.structure_factors is None)

    def test_setup_diffracting_planes(self):

        unit_cell = [5.028, 5.028, 5.519, 90., 90., 120.]
        sgname = "P3221"
        ph = Phase(unit_cell, sgname)

        wavelength = 1.0
        min_bragg_angle = 1 * np.pi / 180
        max_bragg_angle = 25 * np.pi / 180

        ph.setup_diffracting_planes(
            wavelength,
            min_bragg_angle,
            max_bragg_angle,
            verbose=False)

        self.assertEqual(ph.miller_indices.shape[1], 3)
        self.assertTrue(ph.miller_indices.shape[0] > 10)

        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = 'Fm-3m'  # Iron
        ph = Phase(unit_cell, sgname)
        ph.setup_diffracting_planes(
            wavelength,
            min_bragg_angle,
            max_bragg_angle,
            verbose=False)
        for i in range(ph.miller_indices.shape[0]):
            hkl = ph.miller_indices[i, :]
            # Only all even or all odd gives diffraction for a cubic crystal
            self.assertTrue(
                (hkl[0] %
                 2 == 0 and hkl[1] %
                 2 == 0 and hkl[2] %
                 2 == 0) or (
                    hkl[0] %
                    2 == 1 and hkl[1] %
                    2 == 1 and hkl[2] %
                    2 == 1))

    def test_set_structure_factors(self):
        data = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                'data'),
            'Fe_mp-150_conventional_standard.cif')
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = 'Fm-3m'  # Iron
        ph = Phase(unit_cell, sgname, path_to_cif_file=data)
        wavelength = 1.0
        min_bragg_angle = 1 * np.pi / 180
        max_bragg_angle = 25 * np.pi / 180
        ph.setup_diffracting_planes(
            wavelength,
            min_bragg_angle,
            max_bragg_angle,
            verbose=False)

        for i in range(ph.structure_factors.shape[0]):
            self.assertGreaterEqual(ph.structure_factors[i, 0], 0)


if __name__ == '__main__':
    unittest.main()
