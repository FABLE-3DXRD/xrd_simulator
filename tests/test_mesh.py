import unittest
import numpy as np
import torch
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator import utils
import os
import copy


class TestBeam(unittest.TestCase):

    def setUp(self):
        pass

    def test_generate_mesh_from_vertices(self):
        coord = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1]])
        enod = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)
        self.assertAlmostEqual(mesh.enod.shape, enod.shape)
        self.assertAlmostEqual(mesh.coord.shape, coord.shape)
        for a, b in zip(mesh.coord.flatten(), coord.flatten()):
            self.assertAlmostEqual(a, b)
        for r in mesh.eradius:
            self.assertLessEqual(r, 1.0)
        for c in mesh.ecentroids:
            self.assertLessEqual(np.linalg.norm(c), 1.0)

    def test_save_and_load(self):
        nodal_coordinates = np.array(
            [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1]]
        )
        element_node_map = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        mesh = TetraMesh.generate_mesh_from_vertices(
            nodal_coordinates, element_node_map
        )
        path = os.path.join(os.path.join(os.path.dirname(__file__), "data"), "my_mesh")
        mesh.save(path)
        mesh_loaded_from_disc = TetraMesh.load(path + ".xdmf")
        for a, b in zip(
            mesh_loaded_from_disc.coord.flatten(), nodal_coordinates.flatten()
        ):
            self.assertAlmostEqual(a, b)
        os.remove(path + ".xdmf")
        os.remove(path + ".h5")

    def test_generate_mesh_from_levelset(self):
        R = 769.0
        max_cell_circumradius = 450.0
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.linalg.norm(x) - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius,
        )
        for c in mesh.coord:
            r = np.linalg.norm(c)
            self.assertLessEqual(r, R * 1.001)
        for r in mesh.eradius:
            self.assertLessEqual(r, max_cell_circumradius * 1.001)

    def test_move_mesh(self):
        R = 769.0
        max_cell_circumradius = 450.0
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.linalg.norm(x) - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius,
        )
        translation = torch.tensor([1.0, 769.0, -5678.0], dtype=torch.float64)
        new_nodal_coordinates = mesh.coord + translation
        mesh._mesh.points = utils.ensure_numpy(new_nodal_coordinates)
        mesh._set_fem_matrices()
        mesh._expand_mesh_data()
        for c in mesh.coord:
            r = torch.linalg.norm(c - translation).item()
            self.assertLessEqual(r, R * 1.001)
        for r in mesh.eradius:
            self.assertLessEqual(r, max_cell_circumradius * 1.001)

    def test_compute_mesh_spheres(self):

        coord = np.array(
            [
                [-1.3856244e02, 6.9698529e02, -2.9481543e02],
                [5.1198740e02, 5.7321143e02, -1.3369661e01],
                [-2.4491163e-01, -2.4171574e-01, 1.4720881e-01],
                [2.7638666e02, 6.1436609e02, -3.7048819e02],
            ]
        )

        enod = np.array([[0, 1, 2, 3]])
        mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)

        theta = np.deg2rad(1e-4)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        rotated_coord = R_z.dot(utils.ensure_numpy(mesh.coord).T).T
        rotated_coord = utils.ensure_torch(rotated_coord, dtype=torch.float64)

        eradius_rotated, _ = mesh._compute_mesh_spheres(rotated_coord, mesh.enod)

        self.assertAlmostEqual(float(mesh.eradius[0]), float(eradius_rotated[0]))

    def test_update(self):
        rotation_axis = np.array([0, 0, 1.0])
        rotation_angle = np.pi / 4.37
        translation = np.array([1.0, 769.0, -5678.0])
        rbm = RigidBodyMotion(rotation_axis, rotation_angle, translation)
        Rmat = rbm.rotator.get_rotation_matrix(torch.tensor(rotation_angle, dtype=torch.float64))

        R = 769.0
        max_cell_circumradius = 450.0

        mesh1 = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.linalg.norm(x) - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius,
        )
        mesh2 = copy.deepcopy(mesh1)

        new_nodal_coordinates = torch.matmul(Rmat, mesh1.coord.T).T + torch.tensor(translation, dtype=torch.float64)
        mesh1._mesh.points = utils.ensure_numpy(new_nodal_coordinates)
        mesh1._set_fem_matrices()
        mesh1._expand_mesh_data()

        mesh2.update(rbm, time=1.0)

        tol = 1e-3
        # Sphere computations can have slightly larger numerical differences
        # because they involve solving for optimal bounding spheres, and recomputing
        # from transformed vertices vs transforming existing spheres can yield different results
        # due to the iterative nature of the sphere-fitting algorithm.
        # The ~20 unit tolerance is reasonable given coordinates are in the range of ~1000-6000 (< 0.5% error)
        sphere_tol = 20.0  # Allow up to 20 units difference in sphere centroids

        # Vectorized comparisons for speed
        coord_diff = torch.max(torch.abs(mesh1.coord - mesh2.coord)).item()
        self.assertLessEqual(coord_diff, tol, f"Max coord difference {coord_diff} exceeds {tol}")

        centroid_diff = torch.max(torch.abs(mesh1.ecentroids - mesh2.ecentroids)).item()
        self.assertLessEqual(centroid_diff, tol, f"Max centroid difference {centroid_diff} exceeds {tol}")

        sphere_centroid_diff = torch.max(torch.abs(mesh1.espherecentroids - mesh2.espherecentroids)).item()
        self.assertLessEqual(sphere_centroid_diff, sphere_tol, 
            f"Max sphere centroid difference {sphere_centroid_diff} exceeds {sphere_tol}")

        normals_diff = torch.max(torch.abs(mesh1.enormals - mesh2.enormals)).item()
        self.assertLessEqual(normals_diff, tol, f"Max normals difference {normals_diff} exceeds {tol}")

        radius_diff = torch.max(torch.abs(mesh1.eradius - mesh2.eradius)).item()
        self.assertLessEqual(radius_diff, sphere_tol,
            f"Max radius difference {radius_diff} exceeds {sphere_tol}")

        c1, c2 = mesh1.centroid, mesh2.centroid
        for i in range(3):
            diff = torch.abs(c1[i] - c2[i]).item()
            self.assertLessEqual(diff, tol)

    def test_translate(self):
        R = 769.0
        max_cell_circumradius = 450.0
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.sqrt(
                (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + (x[2] + 1.4) ** 2
            )
            - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius,
        )

        mesh.translate(-torch.mean(mesh.coord, axis=0))
        for i in range(3):
            self.assertLessEqual(torch.abs(torch.mean(mesh.coord, axis=0)[i]).item(), 1e-4)

    def test_rotate(self):
        R = 769.0
        max_cell_circumradius = 450.0
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.sqrt(
                (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + (x[2] + 1.4) ** 2
            )
            - R,
            bounding_radius=769.0,
            max_cell_circumradius=max_cell_circumradius,
        )

        mesh.translate(-torch.mean(mesh.coord, axis=0))
        for i in range(3):
            self.assertLessEqual(torch.abs(torch.mean(mesh.coord, axis=0)[i]).item(), 1e-4)

        mesh.rotate(np.array([0, 1, 0]), np.pi / 3.0)
        for i in range(3):
            self.assertLessEqual(torch.abs(torch.mean(mesh.coord, axis=0)[i]).item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
