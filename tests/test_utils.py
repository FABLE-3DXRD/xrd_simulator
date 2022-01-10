import unittest
import numpy as np
from xrd_simulator.xfab import tools
from xrd_simulator import utils
from scipy.spatial.transform import Rotation


class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)  # changes all randomisation in the test

    def test_clip_line_with_convex_polyhedron(self):
        line_points = np.ascontiguousarray([[-1., 0.2, 0.2], [-1., 0.4, 0.6]])
        line_direction = np.ascontiguousarray([1.0, 0.0, 0.0])
        line_direction = line_direction / np.linalg.norm(line_direction)
        plane_points = np.ascontiguousarray([[0., 0.5, 0.5], [1, 0.5, 0.5], [0.5, 0.5, 0.], [
                                            0.5, 0.5, 1.], [0.5, 0, 0.5], [0.5, 1., 0.5]])
        plane_normals = np.ascontiguousarray(
            [[-1., 0., 0.], [1., 0., 0.], [0., 0., -1.], [0., 0., 1.], [0., -1., 0.], [0., 1., 0.]])
        clip_lengths = utils.clip_line_with_convex_polyhedron(
            line_points, line_direction, plane_points, plane_normals)
        for clip_length in clip_lengths:
            self.assertAlmostEqual(
                clip_length,
                1.0,
                msg="Projection through unity cube should give unity clip length")

        line_direction = np.ascontiguousarray([1.0, 0.2, 0.1])
        line_direction = line_direction / np.linalg.norm(line_direction)
        clip_lengths = utils.clip_line_with_convex_polyhedron(
            line_points, line_direction, plane_points, plane_normals)
        for clip_length in clip_lengths:
            self.assertGreater(
                clip_length,
                1.0,
                msg="Tilted projection through unity cube should give greater than unity clip length")

    def test_lab_strain_to_B_matrix(self):

        U = Rotation.random().as_matrix()
        strain_tensor = (np.random.rand(3, 3) - 0.5) / \
            100.  # random small strain tensor
        strain_tensor = (strain_tensor.T + strain_tensor) / 2.
        unit_cell = [5.028, 5.028, 5.519, 90., 90., 120.]
        B = utils.lab_strain_to_B_matrix(strain_tensor, U, unit_cell)

        n_c = np.random.rand(3,)  # crystal unit vector
        n_c = n_c / np.linalg.norm(n_c)
        n_l = np.dot(U, n_c)  # lab unit vector

        # strain along n_l described in lab frame
        strain_l = np.dot(np.dot(n_l, strain_tensor), n_l)
        s = tools.b_to_epsilon(B, unit_cell)
        crystal_strain = np.array(
            [[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]])

        # strain along n_l described in crystal frame
        strain_c = np.dot(np.dot(n_c, crystal_strain), n_c)

        # The strain should be invariant along a direction
        self.assertAlmostEqual(
            strain_l,
            strain_c,
            msg="bad crystal to lab frame conversion")

    def test_alpha_to_quarternion(self):
        _, alpha_2, alpha_3 = np.random.rand(3,)
        q = utils.alpha_to_quarternion(0, alpha_2, alpha_3)
        self.assertAlmostEqual(q[0], 1.0, msg="quarternion wrongly computed")
        self.assertAlmostEqual(q[1], 0.0, msg="quarternion wrongly computed")
        self.assertAlmostEqual(q[2], 0.0, msg="quarternion wrongly computed")
        self.assertAlmostEqual(q[3], 0.0, msg="quarternion wrongly computed")
        alpha_1 = np.random.rand(7,)
        alpha_2 = np.random.rand(7,)
        alpha_3 = np.random.rand(7,)
        qq = utils.alpha_to_quarternion(alpha_1, alpha_2, alpha_3)
        for q in qq:
            self.assertTrue(
                np.abs(
                    np.linalg.norm(q) -
                    1.0) < 1e-5,
                msg="quarternion not normalised")

    def test_diffractogram(self):
        diffraction_pattern = np.zeros((20, 20))
        R = 8
        det_c_z, det_c_y = 10., 10.
        for i in range(diffraction_pattern.shape[0]):
            for j in range(diffraction_pattern.shape[1]):
                if np.abs(np.sqrt((i - det_c_z)**2 +
                          (j - det_c_y)**2) - R) < 0.5:
                    diffraction_pattern[i, j] += 1
        bin_centres, histogram = utils.diffractogram(
            diffraction_pattern, det_c_z, det_c_y, 1.0)
        self.assertEqual(
            np.sum(
                histogram > 0),
            1,
            msg="Error in diffractogram azimuth integration")
        self.assertEqual(
            np.sum(histogram),
            np.sum(diffraction_pattern),
            msg="Error in diffractogram azimuth integration")
        self.assertEqual(
            histogram[R],
            np.sum(diffraction_pattern),
            msg="Error in diffractogram azimuth integration")

    def test_get_bounding_ball(self):
        points = np.random.rand(4, 3) - 0.5
        centre, radius = utils._get_bounding_ball(points)
        mean = np.mean(points, axis=0)
        base_radius = np.max(np.linalg.norm(points - mean, axis=1))
        self.assertLessEqual(
            radius,
            base_radius,
            msg="Ball is larger than initial guess")

        for p in points:
            self.assertLessEqual(
                (p - centre[0:3]).dot(p - centre[0:3]),
                (radius * 1.0001)**2,
                msg="Point not contained by ball")

        ratios = []
        for _ in range(500):
            points = np.random.rand(4, 3) - 0.5
            centre, radius = utils._get_bounding_ball(points)
            mean = np.mean(points, axis=0)
            base_radius = np.max(np.linalg.norm(points - mean, axis=1))
            ratios.append(radius / base_radius)
        self.assertLessEqual(
            np.mean(ratios),
            0.9,
            msg="Averag radius decrease less than 10%")


if __name__ == '__main__':
    unittest.main()
