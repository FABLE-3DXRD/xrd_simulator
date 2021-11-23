import unittest
import numpy as np
from xrd_simulator.detector import Detector
from xrd_simulator.phase import Phase
from xrd_simulator.scatterer import Scatterer
from scipy.spatial import ConvexHull
from xrd_simulator.beam import Beam
import os
import matplotlib.pyplot as plt

class TestDetector(unittest.TestCase):

    def setUp(self):
        self.pixel_size = 50.
        self.detector_size = 10000.
        self.d0 = np.array([1,0,0])*self.detector_size
        self.d1 = np.array([1,1,0])*self.detector_size
        self.d2 = np.array([1,0,1])*self.detector_size
        self.detector = Detector( self.pixel_size, self.d0, self.d1, self.d2 )

    def test_init(self):

        for o,otrue in zip(self.detector.d0, np.array([1,0,0])*self.detector_size):
            self.assertAlmostEqual(o,otrue ,msg="detector origin is incorrect")

        for z,ztrue in zip(self.detector.zdhat, np.array([0,0,1])):
            self.assertAlmostEqual(z, ztrue ,msg="zdhat is incorrect")

        for y,ytrue in zip(self.detector.ydhat, np.array([0,1,0])):
            self.assertAlmostEqual(y, ytrue ,msg="ydhat is incorrect")

        self.assertAlmostEqual(self.detector.zmax, self.detector_size, msg="Bad detector dimensions in zmax")
        self.assertAlmostEqual(self.detector.ymax, self.detector_size, msg="Bad detector dimensions in ymax")

        for n,ntrue in zip(self.detector.normal, np.array([-1,0,0])):
            self.assertAlmostEqual(n, ntrue ,msg="Bad detector normal")

    def test_centroid_render(self):
        v = self.detector.ydhat + self.detector.zdhat
        v = v/np.linalg.norm(v)
        verts1 = np.array([[0,0,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0]]) + v*np.sqrt(2)*self.detector_size/2. # tetra at detector centre
        ch1 = ConvexHull( verts1 )
        verts2 = np.array([[0,0,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0]]) + 2*v*np.sqrt(2)*self.detector_size # tetra out of detector bounds
        ch2 = ConvexHull( verts2 )
        wavelength = 1.0

        incident_wave_vector = 2*np.pi* np.array([1,0,0])/(wavelength)
        scattered_wave_vector = self.d0 + self.pixel_size*3*self.detector.ydhat + self.pixel_size*2*self.detector.zdhat
        scattered_wave_vector = 2*np.pi*scattered_wave_vector/(np.linalg.norm(scattered_wave_vector)*wavelength)

        data = os.path.join( os.path.join(os.path.dirname(__file__), 'data' ), 'Fe_mp-150_conventional_standard.cif' )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = 'Fm-3m' # Iron
        phase   = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase.setup_diffracting_planes(wavelength, 0, 20*np.pi/180)

        scatterer1 = Scatterer( ch1, 
                                scattered_wave_vector=scattered_wave_vector, 
                                incident_wave_vector=incident_wave_vector, 
                                wavelength=wavelength,
                                incident_polarization_vector=np.array([0,1,0]), 
                                rotation_axis=np.array([0,0,1]),
                                time=0, 
                                phase=phase, 
                                hkl_indx=0 )
        scatterer2 = Scatterer( ch2, 
                                scattered_wave_vector=scattered_wave_vector, 
                                incident_wave_vector=incident_wave_vector, 
                                wavelength=wavelength,
                                incident_polarization_vector=np.array([0,1,0]), 
                                rotation_axis=np.array([0,0,1]),
                                time=0, 
                                phase=phase, 
                                hkl_indx=0 )

        self.detector.frames.append([scatterer1, scatterer2])
        piximage = self.detector.render(frame_number=0, lorentz=False, polarization=False, structure_factor=False, method="centroid")
        npix = int( self.detector_size / (2*self.pixel_size) )
        self.assertAlmostEqual(piximage[npix+2,npix+3], ch1.volume, msg="detector rendering did not capture scatterer")
        self.assertAlmostEqual(np.sum(piximage), ch1.volume, msg="detector rendering captured out of bounds scatterer")

        # Try rendering with advanced intensity model
        piximage = self.detector.render(frame_number=0, lorentz=True, polarization=False, structure_factor=False)
        self.assertTrue(piximage[npix+2,npix+3]!=ch1.volume, msg="detector rendering did not use lorentz factor")
        piximage = self.detector.render(frame_number=0, lorentz=False, polarization=True, structure_factor=False)
        self.assertTrue(piximage[npix+2,npix+3]!=ch1.volume, msg="detector rendering did not use polarization factor")
        piximage = self.detector.render(frame_number=0, lorentz=False, polarization=False, structure_factor=True)
        self.assertTrue(piximage[npix+2,npix+3]!=ch1.volume, msg="detector rendering did not use structure_factor factor")

    def test_projection_render(self):
        
        # Convex hull of a sphere placed at the centre of the detector
        phi, theta  = np.meshgrid( np.linspace( 0, 2*np.pi, 25 ), np.linspace( 0, 2*np.pi, 25 ), indexing='ij' )
        r     = 1.0*self.detector_size/4.
        x,y,z = r*np.cos( phi )*np.sin( theta ),r*np.sin( phi )*np.sin( theta ),r*np.cos( theta )
        hull_points = np.array( [x.flatten(),y.flatten(),z.flatten()] ).T
        v = self.detector.ydhat + self.detector.zdhat
        v = v/np.linalg.norm(v)
        sphere_hull = ConvexHull( hull_points + v*np.sqrt(2)*self.detector_size/2. )

        # The spherical scatterer forward scatters.
        wavelength = 1.0
        incident_wave_vector = 2*np.pi* np.array([1,0,0])/wavelength
        scattered_wave_vector = 2*np.pi* np.array([1,0,0])/wavelength
        scattered_wave_vector = 2*np.pi*scattered_wave_vector/(np.linalg.norm(scattered_wave_vector)*wavelength)

        # The spherical scatterer is composed of Fe_mp-150 (a pure iron crystal)
        data = os.path.join( os.path.join(os.path.dirname(__file__), 'data' ), 'Fe_mp-150_conventional_standard.cif' )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = 'Fm-3m' # Iron
        phase   = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase.setup_diffracting_planes(wavelength, 0, 20*np.pi/180)

        scatterer = Scatterer(  sphere_hull, 
                                scattered_wave_vector=scattered_wave_vector, 
                                incident_wave_vector=incident_wave_vector, 
                                wavelength=wavelength,
                                incident_polarization_vector=np.array([0,1,0]), 
                                rotation_axis=np.array([0,0,1]),
                                time=0, 
                                phase=phase,
                                hkl_indx=0 )

        self.detector.frames.append([scatterer])
        piximage = self.detector.render(frame_number=0, lorentz=False, polarization=False, structure_factor=False, method="project")

        projected_summed_intensity = np.sum(piximage)*(self.pixel_size**2)
        relative_error             = np.abs( scatterer.volume - projected_summed_intensity ) / scatterer.volume
        self.assertLessEqual( relative_error, 1e-4, msg="Projected mass does not match the hull volume" )

        index = np.where(piximage==np.max(piximage))
        self.assertEqual( index[0][0], self.detector_size//(2*self.pixel_size), msg="Projected mass does not match the hull volume" )
        self.assertEqual( index[1][0], self.detector_size//(2*self.pixel_size), msg="Projected mass does not match the hull volume" )

        no_pixels = int( self.detector_size//self.pixel_size )
        print("no_pixels", no_pixels)
        for i in range(no_pixels):
            for j in range(no_pixels):
                if (i - no_pixels/2.)**2 + (j - no_pixels/2.)**2 > (1.01*r/self.pixel_size)**2:
                    self.assertAlmostEqual( piximage[i,j], 0 )
                elif (i - no_pixels/2.)**2 + (j - no_pixels/2.)**2 < (0.99*r/self.pixel_size)**2:
                    self.assertGreater( piximage[i,j], 0 )

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

if __name__ == '__main__':
    unittest.main()