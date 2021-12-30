import unittest
import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xfab import tools
from scipy.signal import convolve
import os

class TestPolycrystal(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)

        self.pixel_size = 75.
        self.detector_size = self.pixel_size*1024
        self.detector_distance = 142938.28756189224
        self.d0 = np.array([self.detector_distance,   -self.detector_size/2.,  -self.detector_size/2.])
        self.d1 = np.array([self.detector_distance,    self.detector_size/2.,  -self.detector_size/2.])
        self.d2 = np.array([self.detector_distance,   -self.detector_size/2.,   self.detector_size/2.])

        self.detector = Detector( self.pixel_size, self.pixel_size, self.d0, self.d1, self.d2 )

        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set = lambda x: np.dot( x, x ) - self.detector_size/10.,
            bounding_radius = 1.1*self.detector_size/10., 
            max_cell_circumradius = 0.01*self.detector_size/10. )

        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221' # Quartz
        phases = [Phase(unit_cell, sgname)]
        B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
        eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )
        euler_angles = np.random.rand(mesh.number_of_elements, 3) * 2 * np.pi
        eU = np.array( [tools.euler_to_u(ea[0], ea[1], ea[2]) for ea in euler_angles] )
        ephase = np.zeros((mesh.number_of_elements,)).astype(int)
        self.polycrystal = Polycrystal(mesh, ephase, eU, eB, phases)

        w = self.detector_size/2. # full field illumination
        beam_vertices = np.array([
            [-self.detector_distance, -w, -w ],
            [-self.detector_distance,  w, -w ],
            [-self.detector_distance,  w,  w ],
            [-self.detector_distance, -w,  w ],
            [ self.detector_distance, -w, -w  ],
            [ self.detector_distance,  w, -w  ],
            [ self.detector_distance,  w,  w  ],
            [ self.detector_distance, -w,  w  ]])
        wavelength = 0.285227
        xray_propagation_direction = np.array([1,0,0]) * 2 * np.pi / wavelength
        polarization_vector = np.array([0,1,0])
        self.beam = Beam(beam_vertices, xray_propagation_direction, wavelength, polarization_vector)

    def test_diffract(self):

        rotation_angle = 10*np.pi/180.
        rotation_axis = np.array([0,0,1])
        translation = np.array([0,0,0])
        motion  = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        self.polycrystal.diffract( self.beam, self.detector, motion )

        # The rendered diffraction pattern should have intensity
        diffraction_pattern = self.detector.render(frame_number=0, lorentz=True, polarization=True, structure_factor=True)

        self.assertGreater(np.sum(diffraction_pattern), 0)

        # .. and the intensity should be scattered over the image
        w = int(self.detector_size/5.)
        for i in range(w, diffraction_pattern.shape[0]-w, w ):
            for j in range(w, diffraction_pattern.shape[0]-w, w ):
                subsum = np.sum( diffraction_pattern[i-w:i+w,j-w:j+w] )
                self.assertGreaterEqual(subsum, 0)

        # Scatterers should be confined to rings
        bragg_angles = []
        for scatterer in self.detector.frames[0]:
            kprime = scatterer.scattered_wave_vector
            k = scatterer.incident_wave_vector
            normfactor = np.linalg.norm(k)*np.linalg.norm(kprime)
            tth = np.arccos( np.dot(k, kprime)/normfactor )
            bragg_angles.append(tth/2.)

        # count the number of non-overlaing rings there should be quite a few.
        bragg_angles =  np.degrees( bragg_angles )
        hist = np.histogram(bragg_angles, bins=np.linspace(0,20,360))[0]
        csequence, nosequences = 0, 0
        for i in range(hist.shape[0]):
            if hist[i]>0:
                csequence +=1
            elif csequence>=1:
                    nosequences +=1
                    csequence = 0
        self.assertGreaterEqual(nosequences, 20, msg="Few or no rings appeared from diffraction.")

    def test_save_and_load(self):
        eB = self.polycrystal.eB.copy()
        path = os.path.join( os.path.join(os.path.dirname(__file__), 'data' ), 'my_polycrystal' )
        self.polycrystal.save( path )
        self.polycrystal  = Polycrystal.load( path )
        self.assertTrue( np.allclose( eB, self.polycrystal.eB ), msg='Data corrupted on save and load' )
        os.remove( path )

if __name__ == '__main__':
    unittest.main()