import unittest
import torch
import numpy as np  # Still needed for ConvexHull
from xrd_simulator.beam import Beam
from scipy.spatial import ConvexHull
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.utils import ensure_torch

torch.set_default_dtype(torch.float64)


class TestBeam(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(10)
        self.beam_vertices = ensure_torch(
            [
                [-5.0, 0.0, 0.0],
                [-5.0, 1.0, 0.0],
                [-5.0, 0.0, 1.0],
                [-5.0, 1.0, 1.0],
                [5.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
                [5.0, 0.0, 1.0],
                [5.0, 1.0, 1.0],
            ]
        )
        self.wavelength = torch.rand(1) * 0.5
        self.xray_propagation_direction = torch.rand(3)
        self.xray_propagation_direction = self.xray_propagation_direction / torch.norm(
            self.xray_propagation_direction
        )
        self.polarization_vector = torch.rand(3)
        self.polarization_vector = (
            self.polarization_vector
            - torch.dot(self.polarization_vector, self.xray_propagation_direction)
            * self.xray_propagation_direction
        )
        self.beam = Beam(
            self.beam_vertices,
            self.xray_propagation_direction,
            self.wavelength,
            self.polarization_vector,
        )

    def test_contains(self):
        is_contained = self.beam.contains(self.beam.centroid)
        self.assertTrue(
            is_contained.item() == 1,
            msg="Beam centroid appears to not be contained by beam.",
        )

        is_contained = self.beam.contains(self.beam.centroid + 5.5)
        self.assertTrue(
            is_contained.item() == 0,
            msg="Beam centroid appears to not be contained by beam.",
        )

        points = torch.rand(3, 10) * 0.1 + 0.5
        for ic in self.beam.contains(points):
            self.assertTrue(
                ic.item() == 1,
                msg="Expected interior point of beam is not contained by beam.",
            )

        points[0, :] = 100
        for ic in self.beam.contains(points):
            self.assertTrue(
                ic.item() == 0,
                msg="Expected exterior point of beam appears to be contained by beam.",
            )

    def test_intersect(self):
        vertices = self.beam_vertices
        ch = self.beam.intersect(vertices.numpy())  # ConvexHull needs numpy array
        for i in range(self.beam.vertices.shape[0]):
            v1 = ensure_torch(ch.points[i, :])
            self.assertAlmostEqual(
                torch.min(torch.norm(v1 - self.beam.vertices, dim=1)).item(), 0
            )

        vertices = self.beam_vertices.clone()
        vertices[:, 0] = vertices[:, 0] / 2.0
        ch = self.beam.intersect(vertices.numpy())
        self.assertAlmostEqual(ch.volume, 5)

        # polyhedra completely contained by the beam
        random_points = (torch.rand(25, 3) / 2.0 + 0.1).numpy()
        ch1 = ConvexHull(random_points)
        vertices = ensure_torch(ch1.points[ch1.vertices])
        ch = self.beam.intersect(vertices.numpy())
        self.assertAlmostEqual(ch.volume, ch1.volume)

        # polyhedra completely outside the beam
        random_points = (torch.rand(25, 3) / 2.0 + 20.0).numpy()
        ch1 = ConvexHull(random_points)
        vertices = ensure_torch(ch1.points[ch1.vertices])
        ch = self.beam.intersect(vertices.numpy())
        self.assertTrue(ch is None)

    def test__find_feasible_point(self):
        voxel_verts = ensure_torch(
            [
                [-1.0, 0.0, 0.0],
                [-1.0, 10.0, 0.0],
                [-1.0, 0.0, 10.0],
                [-1.0, 10.0, 10.0],
                [1.0, 0.0, 0.0],
                [1.0, 10.0, 0.0],
                [1.0, 0.0, 10.0],
                [1.0, 10.0, 10.0],
            ]
        )

        # ConvexHull still needs numpy
        poly_halfspace = ConvexHull(voxel_verts.numpy()).equations
        poly_halfspace = ensure_torch(
            np.unique(poly_halfspace.round(decimals=6), axis=0)
        )
        combined_halfspaces = torch.vstack((poly_halfspace, self.beam.halfspaces))
        point = self.beam._find_feasible_point(combined_halfspaces)
        self.assertTrue(point is not None)

    def test__get_proximity_intervals(self):
        self.beam_vertices = ensure_torch(
            [
                [-500.0, -1.0, -1.0],
                [-500.0, -1.0, 1.0],
                [-500.0, 1.0, 1.0],
                [-500.0, 1.0, -1.0],
                [500.0, -1.0, -1.0],
                [500.0, -1.0, 1.0],
                [500.0, 1.0, 1.0],
                [500.0, 1.0, -1.0],
            ]
        )
        self.beam_vertices[:, 1:] = self.beam_vertices[:, 1:] / 10000000.0
        self.xray_propagation_direction = ensure_torch([1.0, 0.0, 0.0])
        self.polarization_vector = torch.rand(3)
        self.polarization_vector = (
            self.polarization_vector
            - torch.dot(self.polarization_vector, self.xray_propagation_direction)
            * self.xray_propagation_direction
        )
        self.beam = Beam(
            self.beam_vertices,
            self.xray_propagation_direction,
            self.wavelength,
            self.polarization_vector,
        )

        rotation_axis = ensure_torch([0.0, 0.0, 1.0])
        rotation_angle = ensure_torch(torch.pi - 1e-8)
        translation = ensure_torch([0.0, 0.0, 0.0])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        sphere_centres = ensure_torch([[400.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
        sphere_radius = ensure_torch([[2.0], [0.5]])

        intervals = self.beam._get_proximity_intervals(
            sphere_centres, sphere_radius, motion
        )

        self.assertEqual(
            len(intervals[0]), 2, msg="Wrong number of proximity intervals"
        )
        self.assertEqual(
            len(intervals[1]), 2, msg="Wrong number of proximity intervals"
        )

        tol = 2.0 / torch.rad2deg(motion.rotation_angle)
        for i in range(sphere_centres.shape[0]):
            fraction_before_beam_leaves_sphere = (
                torch.atan(sphere_radius[i] / torch.norm(sphere_centres[i])) / torch.pi
            )
            self.assertAlmostEqual(
                intervals[i][0][0].item(), 0, msg="Proximity interval wrong"
            )
            self.assertLessEqual(
                torch.abs(
                    intervals[i][0][1] - fraction_before_beam_leaves_sphere[0]
                ).item(),
                tol,
                msg="Proximity interval wrong",
            )

    def test_set_beam_vertices(self):
        new_vertices = ensure_torch(
            [[-5.0, 0.0, 0.0], [-5.0, 1.0, 0.0], [-5.0, 0.0, 1.0], [5.0, 0.0, 0.0]]
        )
        self.beam.set_beam_vertices(new_vertices)
        self.assertTrue(
            torch.allclose(torch.mean(new_vertices, dim=0), self.beam.centroid),
            msg="centroid incorrect",
        )
        self.assertTrue(
            torch.allclose(
                ensure_torch(self.beam.halfspaces.shape), ensure_torch([4, 4])
            )
        )
        self.assertTrue(torch.allclose(self.beam.vertices, new_vertices))

    def test__get_candidate_spheres(self):
        rotation_angle = ensure_torch(10 * torch.pi / 180.0)
        rotation_axis = ensure_torch([0.0, 0.0, 1.0])
        translation = ensure_torch([3.0, 2.0, 1.0])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        sphere_centres = torch.rand(10, 3)
        sphere_centres[-1, :] = ensure_torch([100.0, 100.0, 100.0])
        sphere_radius = torch.rand(10)
        mask, _ = self.beam._get_candidate_spheres(
            sphere_centres, sphere_radius, motion
        )
        mask = torch.sum(mask, dim=0) > 0

        self.assertTrue(torch.all(mask[0:9]))
        self.assertFalse(mask[9].item())


if __name__ == "__main__":
    unittest.main()
