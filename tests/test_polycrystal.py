import unittest
import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xfab import tools

class TestPolycrystal(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)

        self.pixel_size = 75.
        self.detector_size = self.pixel_size*1024
        self.detector_distance = 142938.28756189224
        geometry_matrix_0 = np.array([[ self.detector_distance,0,0],[ self.detector_distance,self.detector_size,0],[self.detector_distance,0,self.detector_size]]).T
        def geometry_matrix(s):
            sin = np.sin( -s*np.pi/2. )
            cos = np.cos( -s*np.pi/2. )
            Rz = np.array([ [ cos, -sin, 0 ],
                            [ sin,  cos, 0 ],
                            [  0,    0,  1 ] ])
            return Rz.dot( geometry_matrix_0 )
        self.detector = Detector( self.pixel_size, geometry_matrix )

        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set = lambda x: np.dot( x, x ) - self.detector_size/10.,
            bounding_radius = 1.1*self.detector_size/10., 
            cell_size = 0.75*self.detector_size/10. )
        mesh.move( -(self.detector.ydhat + self.detector.zdhat)*self.detector_size/2. )
        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221' # Quartz
        phases = [Phase(unit_cell, sgname)]
        B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
        eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )
        euler_angles = np.random.rand(mesh.number_of_elements, 3) * 2 * np.pi
        eU = np.array( [tools.euler_to_u(ea[0], ea[1], ea[2]) for ea in euler_angles] )
        ephase = np.zeros((mesh.number_of_elements,)).astype(int)
        self.polycrystal = Polycrystal(mesh, ephase, eU, eB, phases)

        source_point  = (self.detector.zdhat + self.detector.ydhat) *self.detector_size/2. 
        beam_vertices = np.array([
            [-self.detector_distance, 0.,                 0.                 ],
            [-self.detector_distance, 1024*self.pixel_size, 0.                 ],
            [-self.detector_distance, 0.,                 1024*self.pixel_size ],
            [-self.detector_distance, 1024*self.pixel_size, 1024*self.pixel_size ],
            [ self.detector_distance, 0.,                 0.                 ],
            [ self.detector_distance, 1024*self.pixel_size, 0.                 ],
            [ self.detector_distance, 0.,                 1024*self.pixel_size ],
            [ self.detector_distance, 1024*self.pixel_size, 1024*self.pixel_size ]]) + source_point
        self.beam = Beam(beam_vertices, wavelength=0.285227, k1=np.array([1,0,0]), k2=np.array([0,-1,0]))

    def test_get_candidate_elements(self):
        pass

    def test_diffract(self):
        self.polycrystal.diffract( self.beam, self.detector )
        print("Hej")
        pixim = self.detector.render(frame_number=0)
        print(np.sum(pixim))
        import matplotlib.pyplot as plt
        plt.imshow(pixim)
        plt.show()

if __name__ == '__main__':
    unittest.main()