import unittest
import numpy as np
from xrd_simulator.beam import Beam
from scipy.spatial import ConvexHull

class TestBeam(unittest.TestCase):

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test
        self.beam_vertices = np.array([
            [-5., 0., 0. ],
            [-5., 1., 0. ],
            [-5., 0., 1. ],
            [-5., 1., 1. ],
            [ 5., 0., 0. ],
            [ 5., 1., 0. ],
            [ 5., 0., 1. ],
            [ 5., 1., 1. ]])
        self.wavelength = self.get_pseudorandom_wavelength()
        self.k1, self.k2 = self.get_pseudorandom_k1_k2(self.wavelength)
        while( np.arccos(  self.k1.dot(self.k2)/(np.linalg.norm( self.k1,)*np.linalg.norm(self.k2)) ) > np.pi/4. ):
            self.k1, self.k2 = self.get_pseudorandom_k1_k2(self.wavelength)
        self.beam = Beam(self.beam_vertices, self.wavelength, self.k1, self.k2)

    def test_init(self):
        for i in range(3):
            self.assertAlmostEqual( self.beam.k[i], self.k1[i], msg="Initial wavevector not equal to input k1 wavevector" )
        self.assertTrue( np.allclose( self.beam.original_vertices, self.beam_vertices) )

    def test_set_geometry(self):
        self.beam.set_geometry(s=1)
        for i in range(3):
            self.assertAlmostEqual( self.beam.k[i], self.k2[i], msg="Initial wavevector not equal to input k1 wavevector" )

        for i in range(self.beam.vertices.shape[0]):
            v1 = self.beam.vertices[i,:]
            v2 = self.beam.original_vertices[i,:]
            for j in range(v1.shape[0]):
                self.assertNotAlmostEqual(v1[j], v2[j])

        self.beam.set_geometry(s=0.5)
        halfalpha = np.arccos( self.beam.k.dot(self.k1)/(np.linalg.norm(self.beam.k)*np.linalg.norm(self.k1)) ) 
        self.assertAlmostEqual( self.beam.rotator.alpha / 2., halfalpha)

        self.beam.set_geometry(s=0.0)
        for i in range(3):
            self.assertAlmostEqual( self.beam.k[i], self.k1[i], msg="Initial wavevector not equal to input k1 wavevector" )

        self.assertTrue(np.allclose(self.beam.vertices, self.beam_vertices))
    def test_intersect(self):
        vertices = self.beam_vertices
        ch = self.beam.intersect( vertices )
        for i in range(self.beam.vertices.shape[0]):
            v1 = ch.points[i,:]
            v2 = self.beam.original_vertices[i,:]
            self.assertAlmostEqual(  np.min( np.linalg.norm(v1 - self.beam.original_vertices, axis=1) ), 0 )

        vertices = self.beam_vertices
        vertices[:,0] = vertices[:,0]/2. 
        ch = self.beam.intersect( vertices )
        self.assertAlmostEqual(  ch.volume, 5 )

        ch1 = ConvexHull( ( np.random.rand(25,3)/2.+0.1 )  ) # polyhedra completely contained by the beam.
        vertices = ch1.points[ch1.vertices]
        ch = self.beam.intersect( vertices )
        self.assertAlmostEqual(  ch.volume, ch1.volume )

    def get_pseudorandom_wavelength(self):
        return np.random.rand()*0.5

    def get_pseudorandom_k1_k2(self, wavelength):
        k1 = (np.random.rand(3,)-0.5)*2
        k2 = (np.random.rand(3,)-0.5)*2
        k1 = 2*np.pi*k1/(np.linalg.norm(k1)*wavelength)
        k2 = 2*np.pi*k2/(np.linalg.norm(k2)*wavelength)
        return k1, k2

if __name__ == '__main__':
    unittest.main()