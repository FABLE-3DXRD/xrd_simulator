import unittest
import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xfab import tools
from scipy.signal import convolve

class TestPolycrystal(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)
        totrot = 2*np.pi/180
        self.pixel_size = 75.
        self.detector_size = self.pixel_size*1024
        self.detector_distance = 142938.28756189224
        geometry_matrix_0 = np.array([
            [self.detector_distance,   -self.detector_size/2.,  -self.detector_size/2.],
            [self.detector_distance,    self.detector_size/2.,  -self.detector_size/2.],
            [self.detector_distance,   -self.detector_size/2.,   self.detector_size/2.]]).T
        def geometry_descriptor(s):
            sin = np.sin( -s*totrot )
            cos = np.cos( -s*totrot )
            Rz = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])
            return Rz.dot( geometry_matrix_0 )
        self.detector = Detector( self.pixel_size, geometry_descriptor )

        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set = lambda x: np.dot( x, x ) - self.detector_size/10.,
            bounding_radius = 1.1*self.detector_size/10., 
            cell_size = 0.0075*self.detector_size/10)
    
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

        sin = np.sin( -totrot )
        cos = np.cos( -totrot )
        Rz = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])

        k1 = np.array([1,0,0]) * 2 * np.pi / wavelength
        k2 = Rz.dot(k1)
        self.beam = Beam(beam_vertices, wavelength, k1, k2, translation=np.array([0., 0., 0.]))


    def test_diffract(self):

        self.polycrystal.diffract( self.beam, self.detector )

        # The rendered diffraction pattern should have intensity
        pixim = self.detector.render(frame_number=0)
        self.assertGreater(np.sum(pixim), 0)

        # .. and the intensity should be scattered over the image
        kernel = np.ones((300,300))
        pixim = convolve(pixim, kernel, mode='same', method='auto')
        self.assertGreaterEqual(np.sum(pixim>0), pixim.shape[0]*pixim.shape[1])

        # Scatterers should be confined to rings
        bragg_angles = []
        for scatterer in self.detector.frames[0]:
            self.beam.set_geometry(s=scatterer.s)
            normfactor = np.linalg.norm(self.beam.k)*np.linalg.norm(scatterer.kprime)
            tth = np.arccos( np.dot(self.beam.k, scatterer.kprime)/normfactor )
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


if __name__ == '__main__':
    unittest.main()