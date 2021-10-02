import unittest
import numpy as np
from xrd_simulator import beam
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
        self.translation = np.array([0.,0.,0.])
        self.beam = Beam(self.beam_vertices, self.wavelength, self.k1, self.k2, self.translation)

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

    def test_translation(self):
        self.translation = np.random.rand(3,)
        self.beam = Beam(self.beam_vertices, self.wavelength, self.k1, self.k2, self.translation)
        self.beam.set_geometry(s=1)
        for i in range(self.beam.vertices.shape[0]):
            v1 = self.beam.rotator( self.beam.original_vertices[i] + self.translation, s=1 ) 
            v2 = self.beam.vertices[i]
            for k in range(3):
                self.assertAlmostEqual(v1[k], v2[k], msg='Beam vertices are not properly translated')

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

        ch1 = ConvexHull( ( np.random.rand(25,3)/2.+20.0 )  ) # polyhedra completely outside the beam.
        vertices = ch1.points[ch1.vertices]
        ch = self.beam.intersect( vertices )
        self.assertTrue(  ch is None )

    def test_get_proximity_intervals(self):
        sphere_centre = np.array([20.0, 0.0, 0.0])
        sphere_radius = 1.0
        angle1 = np.arctan(sphere_radius/sphere_centre[0])
        self.beam_vertices = np.array([
            [-50., -1., -1. ],
            [-50., -1.,  1. ],
            [-50.,  1.,  1. ],
            [-50.,  1., -1. ],
            [50.,  -1., -1. ],
            [50.,  -1.,  1. ],
            [50.,   1.,  1. ],
            [50.,   1., -1. ]  ])
        self.beam_vertices[:,1:] = self.beam_vertices[:,1:]/1000. # tiny beam cross section
        self.k1 = np.array([  1,  0,     0 ])
        self.k2 = np.array([ -1,  0.001, 0 ])
        self.k1 = (np.pi*2/self.wavelength)*self.k1/np.linalg.norm(self.k1)
        self.k2 = (np.pi*2/self.wavelength)*self.k2/np.linalg.norm(self.k2)
        self.translation = np.array([ 0, 0, 0 ])
        self.beam = Beam(self.beam_vertices, self.wavelength, self.k1, self.k2, self.translation)
        intervals = self.beam.get_proximity_intervals(sphere_centre, sphere_radius)

        self.assertEqual( intervals.shape[0], 2, msg="Wrong number of proximity intervals" )

        # needs change of points on interval is not exactly 10
        points_on_interval = 10.
        self.assertAlmostEqual( intervals[0,0], 0/(points_on_interval-1) , msg="Proximity interval wrong" )
        self.assertAlmostEqual( intervals[0,1], 1/(points_on_interval-1) , msg="Proximity interval wrong" )
        self.assertAlmostEqual( intervals[1,0], 8/(points_on_interval-1) , msg="Proximity interval wrong" )
        self.assertAlmostEqual( intervals[1,1], 9/(points_on_interval-1) , msg="Proximity interval wrong" )

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