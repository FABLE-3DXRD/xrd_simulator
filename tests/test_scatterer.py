import unittest
import numpy as np
from xrd_simulator.scatterer import Scatterer
from scipy.spatial import ConvexHull
from xrd_simulator.phase import Phase
import os


class TestScatterer(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)
        self.wavelength = 1.0
        verts = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0.1, 0.1, 0.1]])
        self.ch = ConvexHull(verts)
        scattered_wave_vector = np.random.rand(3,)
        self.scattered_wave_vector = 2 * np.pi * scattered_wave_vector / \
            (self.wavelength * np.linalg.norm(scattered_wave_vector))
        self.time = np.random.rand()

        data = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                'data'),
            'Fe_mp-150_conventional_standard.cif')
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = 'Fm-3m'  # Iron
        self.phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        self.phase.setup_diffracting_planes(
            self.wavelength, 0, 20 * np.pi / 180, verbose=False)

        self.incident_wave_vector = np.array([1, 0., 0])
        self.incident_wave_vector = 2 * np.pi * self.incident_wave_vector / \
            (self.wavelength * np.linalg.norm(self.incident_wave_vector))
        self.incident_polarization_vector = np.array([0., 1., 0])
        self.rotation_axis = np.array([0., 0, 1.])

        self.scatterer = Scatterer(self.ch,
                                   self.scattered_wave_vector,
                                   self.incident_wave_vector,
                                   self.wavelength,
                                   self.incident_polarization_vector,
                                   self.rotation_axis,
                                   self.time,
                                   self.phase,
                                   hkl_indx=0)

    def test_lorentz(self):
        z = np.array([0, 1, 1.])
        self.scatterer.scattered_wave_vector = 2 * np.pi * \
            z / (np.linalg.norm(z) * self.wavelength)
        L = self.scatterer.lorentz_factor
        self.assertAlmostEqual(L, np.sqrt(2.))

        z = np.array([0, 0, 1.])
        self.scatterer.scattered_wave_vector = 2 * \
            np.pi * z / (self.wavelength)
        L = self.scatterer.lorentz_factor
        self.assertTrue(L is np.inf)

    def test_polarization(self):
        self.scatterer.scattered_wave_vector = np.array([0, 1.0, 0])
        P = self.scatterer.polarization_factor
        self.assertAlmostEqual(P, 0)

        self.scatterer.scattered_wave_vector = np.array(
            [1. / np.sqrt(2), 0, -1. / np.sqrt(2)])
        P = self.scatterer.polarization_factor
        self.assertAlmostEqual(P, 1.0)

    def test_hkl(self):
        hkl = self.scatterer.hkl
        for i in range(3):
            self.assertAlmostEqual(hkl[i], -1, msg="hkl is wrong")

    def test_structure_factor(self):
        structure_factor = self.scatterer.real_structure_factor
        self.assertGreaterEqual(
            structure_factor, 0, msg="structure factor is wrong")

    def test_centroid(self):
        centroidet_corner_2 = self.scatterer.centroid
        for c1, c2 in zip(np.array([0.25, 0.25, 0.25]), centroidet_corner_2):
            self.assertAlmostEqual(c1, c2, msg="centroid is wrong")

    def test_volume(self):
        vol = self.scatterer.volume
        self.assertAlmostEqual(vol, 1 / 6., msg="volume is wrong")


if __name__ == '__main__':
    unittest.main()
