import unittest
import numpy as np
from xrd_simulator.scatterer import Scatterer
from scipy.spatial import ConvexHull
import pkg_resources
from xrd_simulator.phase import Phase
import os

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

        data = os.path.join( os.path.join(os.path.dirname(__file__), 'data' ), 'Fe_mp-150_conventional_standard.cif' )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = 'Fm-3m' # Iron
        self.ph   = Phase(unit_cell, sgname, path_to_cif_file=data)
        self.ph.setup_diffracting_planes(wavelength, 0, 20*np.pi/180)

        bragg_angle = None
        self.scatterer = Scatterer(self.ch, self.kprime, bragg_angle, self.s, self.ph, 0)

    def test_hkl(self):
        hkl = self.scatterer.hkl
        for i in range(3):
            self.assertAlmostEqual( hkl[i], -1, msg="hkl is wrong" )

    def test_structure_factor(self):
        structure_factor = self.scatterer.structure_factor
        self.assertGreaterEqual( structure_factor[0], 0, msg="structure factor is wrong" )

    def test_centroid(self):
        centroid2 = self.scatterer.centroid
        for c1,c2 in zip(np.array([0.25, 0.25, 0.25]), centroid2):
            self.assertAlmostEqual( c1, c2, msg="centroid is wrong" )

    def test_volume(self):
        vol = self.scatterer.volume
        self.assertAlmostEqual( vol, 1/6., msg="volume is wrong" )

if __name__ == '__main__':
    unittest.main()