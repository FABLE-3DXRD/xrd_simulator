import unittest
import numpy as np
from xrd_simulator.detector import Detector
from xrd_simulator.phase import Phase
from xrd_simulator.scatterer import Scatterer
from scipy.spatial import ConvexHull
from xrd_simulator.beam import Beam

class TestDetector(unittest.TestCase):

    def setUp(self):
        # TODO Updat the tests to work in the new labframe
        self.pixel_size = 50.
        self.detector_size = 10000.
        geometry_matrix_0 = np.array([[1,0,0],[1,1,0],[1,0,1]]).T*self.detector_size
        def geometry_descriptor(s):
            sin = np.sin( -s*np.pi/2. )
            cos = np.cos( -s*np.pi/2. )
            Rz = np.array([ [ cos, -sin, 0 ],
                            [ sin,  cos, 0 ],
                            [  0,    0,  1 ] ])
            return Rz.dot( geometry_matrix_0 )
        self.detector = Detector( self.pixel_size, geometry_descriptor )

    def test_init(self):

        for o,otrue in zip(self.detector.detector_origin, np.array([1,0,0])*self.detector_size):
            self.assertAlmostEqual(o,otrue ,msg="detector origin is incorrect")

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

        for o,otrue in zip(self.detector.detector_origin, np.array([0,-1,0])*self.detector_size):
            self.assertAlmostEqual(o,otrue ,msg="detector origin is incorrect")

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
        v = self.detector.ydhat + self.detector.zdhat
        v = v/np.linalg.norm(v)
        verts1 = np.array([[0,0,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0]]) + v*np.sqrt(2)*self.detector_size/2. # cube at detector centre
        ch1 = ConvexHull( verts1 )
        verts2 = np.array([[0,0,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0]]) + 2*v*np.sqrt(2)*self.detector_size # cube out of detector bounds
        ch2 = ConvexHull( verts2 )
        wavelength = 1.0
        kprime = 2*np.pi*np.array([1,0,0])/(wavelength)

        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221' # Quartz
        phase = Phase(unit_cell, sgname)

        bragg_angle = np.random.rand()*np.pi
        scatterer1 = Scatterer(ch1, kprime, bragg_angle, s=0, phase=phase, hkl_indx=0)
        scatterer2 = Scatterer(ch2, kprime, bragg_angle, s=0, phase=phase, hkl_indx=0)
        self.detector.frames.append([scatterer1, scatterer2])
        piximage = self.detector.render(frame_number=0)
        ic = int( (self.detector_size/self.pixel_size) / 2 )
        self.assertAlmostEqual(piximage[ic,ic], ch1.volume, msg="detector rendering did not capture scatterer")
        self.assertAlmostEqual(np.sum(piximage), ch1.volume, msg="detector rendering captured out of bounds scatterer")

    def test_get_intersection(self):

        # central normal algined ray
        ray_direction = np.array([2.23,0.,0.])
        source_point  = np.array([0.,0.,0.])
        z, y  = self.detector.get_intersection(ray_direction, source_point)
        self.assertAlmostEqual(z,0,msg="central detector-normal algined ray does not intersect at 0")
        self.assertAlmostEqual(y,0,msg="central detector-normal algined ray does not intersect at 0")

        # translate the ray
        source_point  += self.detector.ydhat * self.pixel_size
        source_point  -= self.detector.zdhat * 2 * self.pixel_size
        z, y  = self.detector.get_intersection(ray_direction, source_point)
        self.assertAlmostEqual(z, -2*self.pixel_size, msg="translated detector-normal algined ray does not intersect properly")
        self.assertAlmostEqual(y, self.pixel_size, msg="translated detector-normal algined ray does not intersect properly")

        # tilt the ray
        ang = np.arctan(self.pixel_size/self.detector_size)
        frac = np.tan(ang)*np.linalg.norm(ray_direction)
        ray_direction += self.detector.ydhat * frac * 3
        z, y  = self.detector.get_intersection(ray_direction, source_point)
        self.assertAlmostEqual(z, -2*self.pixel_size, msg="translated and tilted ray does not intersect properly")
        self.assertAlmostEqual(y, 4*self.pixel_size, msg="translated and tilted ray does not intersect properly")

    def test_contains(self):
        c1 = self.detector.contains(self.detector_size/10., self.detector_size/5.)
        self.assertTrue(c1, msg="detector does no contain included point")
        c2= self.detector.contains(-self.detector_size/8., self.detector_size/3.)
        self.assertTrue(not c2, msg="detector contain negative points")
        c4= self.detector.contains(self.detector_size*2*self.pixel_size, self.detector_size/374.)
        self.assertTrue(not c4, msg="detector contain out of bounds points")

    def test_get_wrapping_cone(self):
        wavelength = 1.0
        k = 2 * np.pi * np.array([1,0,0]) / wavelength
        source_point  = (self.detector.zdhat + self.detector.ydhat) * self.detector_size/2.
        opening_angle = self.detector.get_wrapping_cone(k, source_point)
        expected_angle = np.arctan( np.sqrt(2)*100.*self.pixel_size / self.detector_size ) / 2.
        self.assertAlmostEqual(opening_angle, expected_angle, msg="detector centered wrapping cone has faulty opening angle")

        source_point  = (self.detector.zdhat + self.detector.ydhat) * self.detector_size/2.
        source_point -= self.detector.zdhat*10*self.pixel_size
        source_point -= self.detector.ydhat*10*self.pixel_size
        opening_angle = self.detector.get_wrapping_cone(k, source_point)
        self.assertGreaterEqual(opening_angle, expected_angle, msg="detector off centered wrapping cone has opening angle")

    def test_approximate_wrapping_cone(self):
        
        source_point  = (10*self.detector.zdhat + 10*self.detector.ydhat) * self.pixel_size
        margin=np.pi/180
        max_expected_angle = margin + np.arctan( ((np.sqrt(2)*self.detector_size)-10.*self.pixel_size) / self.detector_size ) / 2.

        beam_vertices = np.array([
            [-self.detector_size, 0.,                 0.                 ],
            [-self.detector_size, 10*self.pixel_size, 0.                 ],
            [-self.detector_size, 0.,                 10*self.pixel_size ],
            [-self.detector_size, 10*self.pixel_size, 10*self.pixel_size ],
            [ self.detector_size, 0.,                 0.                 ],
            [ self.detector_size, 10*self.pixel_size, 0.                 ],
            [ self.detector_size, 0.,                 10*self.pixel_size ],
            [ self.detector_size, 10*self.pixel_size, 10*self.pixel_size ]]) + source_point
        wavelength = 1.0
        k1 = np.array([1,0,0]) * 2 * np.pi / wavelength
        k2 = np.array([0,-1,0]) * 2 * np.pi / wavelength
        beam = Beam(beam_vertices, wavelength=wavelength, k1=k1, k2=k2, translation=np.array([0,0,0]))

        opening_angle = self.detector.approximate_wrapping_cone( beam, samples=180, margin=margin )
        self.assertGreaterEqual(max_expected_angle, opening_angle, msg="approximated wrapping cone is too large")

if __name__ == '__main__':
    unittest.main()