import unittest
import numpy as np
from xrd_simulator.mesh import TetraMesh
import os

class TestBeam(unittest.TestCase):

    def setUp(self):
        pass

    def test_generate_mesh_from_vertices(self):
        coord = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,-1]])
        enod  = np.array([[0,1,2,3],[0,1,2,4]])
        mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)
        self.assertAlmostEqual(mesh.enod.shape, enod.shape)
        self.assertAlmostEqual(mesh.coord.shape, coord.shape)
        for a,b in zip(mesh.coord.flatten(), coord.flatten()):
            self.assertAlmostEqual(a,b)
        for r in mesh.eradius:
            self.assertLessEqual(r, 1.0)
        for c in mesh.ecentroids:
            self.assertLessEqual(np.linalg.norm(c), 1.0)

    def test_save_and_load(self):
        nodal_coordinates = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,-1]])
        element_node_map  = np.array([[0,1,2,3],[0,1,2,4]])
        mesh = TetraMesh.generate_mesh_from_vertices(nodal_coordinates, element_node_map)
        path = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                'data'),
            'my_mesh')
        mesh.save(path)
        mesh_loaded_from_disc = TetraMesh.load(path + ".xdmf")
        for a,b in zip(mesh_loaded_from_disc.coord.flatten(), nodal_coordinates.flatten()):
            self.assertAlmostEqual(a,b)
        os.remove(path + ".xdmf")
        os.remove(path + ".h5")

    def test_generate_mesh_from_levelset(self):
        R = 769.0
        max_cell_circumradius = 450.
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.linalg.norm(x) - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius)
        for c in mesh.coord:
            r = np.linalg.norm(c)
            self.assertLessEqual(r, R*1.001)
        for r in mesh.eradius:
            self.assertLessEqual(r, max_cell_circumradius*1.001)

    def test_update(self):
        R = 769.0
        max_cell_circumradius = 450.
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.linalg.norm(x) - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius)
        translation = np.array([1.0, 769.0, -5678.0])
        mesh.update( mesh.coord + translation )
        for c in mesh.coord:
            r = np.linalg.norm(c-translation)
            self.assertLessEqual(r, R*1.001)
        for r in mesh.eradius:
            self.assertLessEqual(r, max_cell_circumradius*1.001)

if __name__ == '__main__':
    unittest.main()
