import unittest
import numpy as np
from xrd_simulator.beam import Beam
from scipy.spatial import ConvexHull
from xrd_simulator.motion import RigidBodyMotion

class TestBeam(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)  # changes all randomization in the test
        self.beam_vertices = np.array([
            [-5., 0., 0.],
            [-5., 1., 0.],
            [-5., 0., 1.],
            [-5., 1., 1.],
            [5., 0., 0.],
            [5., 1., 0.],
            [5., 0., 1.],
            [5., 1., 1.]])
        self.wavelength = np.random.rand() * 0.5
        self.xray_propagation_direction = np.random.rand(3,)
        self.xray_propagation_direction = self.xray_propagation_direction / \
            np.linalg.norm(self.xray_propagation_direction)
        self.polarization_vector = np.random.rand(3,)
        self.polarization_vector = self.polarization_vector - \
            np.dot(self.polarization_vector, self.xray_propagation_direction) * self.xray_propagation_direction
        self.beam = Beam(
            self.beam_vertices,
            self.xray_propagation_direction,
            self.wavelength,
            self.polarization_vector)

    def test_contains(self):
        is_contained = self.beam.contains(self.beam.centroid)
        self.assertTrue(is_contained==1,msg='Beam centroid appears to not be contained by beam.')

        is_contained = self.beam.contains(self.beam.centroid + 5.5)
        self.assertTrue(is_contained==0,msg='Beam centroid appears to not be contained by beam.')

        points = np.random.rand(3,10)*0.1 + 0.5
        for ic in self.beam.contains(points):
            self.assertTrue(ic==1,msg='Expected interior point of beam is not contained by beam.')

        points[0,:] = 100
        for ic in self.beam.contains(points):
            self.assertTrue(ic==0,msg='Expected exterior point of beam is appears to be contained by beam.')


    def test_intersect(self):
        vertices = self.beam_vertices
        ch = self.beam.intersect(vertices)
        for i in range(self.beam.vertices.shape[0]):
            v1 = ch.points[i, :]
            self.assertAlmostEqual(
                np.min(
                    np.linalg.norm(
                        v1 -
                        self.beam.vertices,
                        axis=1)),
                0)

        vertices = self.beam_vertices
        vertices[:, 0] = vertices[:, 0] / 2.
        ch = self.beam.intersect(vertices)
        self.assertAlmostEqual(ch.volume, 5)

        # polyhedra completely contained by the beam.
        ch1 = ConvexHull((np.random.rand(25, 3) / 2. + 0.1))
        vertices = ch1.points[ch1.vertices]
        ch = self.beam.intersect(vertices)
        self.assertAlmostEqual(ch.volume, ch1.volume)

        # polyhedra completely outside the beam.
        ch1 = ConvexHull((np.random.rand(25, 3) / 2. + 20.0))
        vertices = ch1.points[ch1.vertices]
        ch = self.beam.intersect(vertices)
        self.assertTrue(ch is None)

    def test__find_feasible_point(self):
        # Corner case when the presolver will find edge interior points that are not clearly inside the 
        # intersection hull.
        voxel_verts = np.array([
            [-1., 0., 0.],
            [-1., 10., 0.],
            [-1., 0., 10.],
            [-1., 10., 10.],
            [1., 0., 0.],
            [1., 10., 0.],
            [1., 0., 10.],
            [1., 10., 10.]])
        poly_halfspace = ConvexHull(voxel_verts).equations
        poly_halfspace = np.unique(poly_halfspace.round(decimals=6), axis=0)
        combined_halfspaces = np.vstack((poly_halfspace, self.beam.halfspaces))
        point = self.beam._find_feasible_point(combined_halfspaces)
        self.assertTrue(point is not None)

    def test__get_proximity_intervals(self):

        self.beam_vertices = np.array([
            [-500., -1., -1.],
            [-500., -1., 1.],
            [-500., 1., 1.],
            [-500., 1., -1.],
            [500., -1., -1.],
            [500., -1., 1.],
            [500., 1., 1.],
            [500., 1., -1.]])
        self.beam_vertices[:, 1:] = self.beam_vertices[:,
                                                       1:] / 10000000.  # tiny beam cross section
        self.xray_propagation_direction = np.array([1, 0, 0])
        self.polarization_vector = np.random.rand(3,)
        self.polarization_vector = self.polarization_vector - \
            np.dot(self.polarization_vector, self.xray_propagation_direction) * self.xray_propagation_direction
        self.beam = Beam(
            self.beam_vertices,
            self.xray_propagation_direction,
            self.wavelength,
            self.polarization_vector)

        rotation_axis = np.array([0., 0., 1.])
        rotation_angle = np.pi - 1e-8
        translation = np.array([0., 0., 0.])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        sphere_centres = np.array([[400.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
        sphere_radius = np.array([[2.0], [0.5]])

        intervals = self.beam._get_proximity_intervals(
            sphere_centres, sphere_radius, motion)

        self.assertEqual(len(intervals[0]), 2,
                         msg="Wrong number of proximity intervals")
        self.assertEqual(len(intervals[1]), 2,
                         msg="Wrong number of proximity intervals")

        tol = 2. / np.degrees(motion.rotation_angle)
        for i in range(sphere_centres.shape[0]):
            # far away sphere small radii approximation:
            fraction_before_beam_leaves_sphere = np.arctan(
                sphere_radius[i] / np.linalg.norm(sphere_centres[i])) / np.pi
            self.assertAlmostEqual(
                intervals[i][0][0], 0, msg="Proximity interval wrong")
            self.assertLessEqual(
                np.abs(intervals[i][0][1] - 
                fraction_before_beam_leaves_sphere[0]), tol,
                msg="Proximity interval wrong")
            self.assertLessEqual(
                np.abs(intervals[i][1][0]-(
                1. - fraction_before_beam_leaves_sphere[0])), tol,
                msg="Proximity interval wrong")
            self.assertAlmostEqual(
                intervals[i][1][1],
                1.0,
                msg="Proximity interval wrong")

        # Now with rotation and translation
        motion.translation = np.array([-87.24, 34.6, 123.34])

        intervals = self.beam._get_proximity_intervals(sphere_centres, sphere_radius, motion)

        self.assertEqual(len(intervals[0]), 1,
                         msg="Wrong number of proximity intervals")
        self.assertEqual(len(intervals[1]), 1,
                         msg="Wrong number of proximity intervals")

        for i in range(sphere_centres.shape[0]):
            # search numerically for the point in time when the sphere leaves
            # the beam and compare to analytical intersection:
            times = np.linspace(0., 1., 10000)
            cs = np.array([motion(sphere_centres[i], t) for t in times])
            L = np.sqrt(cs[:, 1]**2 + cs[:, 2]**2)
            fraction_before_beam_leaves_sphere = times[np.argmin(
                np.abs(L - sphere_radius[i]))]
            self.assertAlmostEqual(
                intervals[i][0][0], 0, msg="Proximity interval wrong")
            self.assertTrue(
                np.abs(
                    intervals[i][0][1] -
                    fraction_before_beam_leaves_sphere) < tol,
                msg="Proximity interval wrong")


    def test_set_beam_vertices(self):
        new_vertices = np.array([[-5., 0., 0.],
                                 [-5., 1., 0.],
                                 [-5., 0., 1.],
                                 [5., 0., 0.]])
        self.beam.set_beam_vertices(new_vertices)
        self.assertTrue( np.allclose(np.mean(new_vertices,axis=0), self.beam.centroid), msg="centroid incorrect")
        self.assertTrue( np.allclose(self.beam.halfspaces.shape, (4,4)))
        self.assertTrue( np.allclose(self.beam.vertices, new_vertices))

    def test__get_candidate_spheres(self):
        rotation_angle = 10 * np.pi / 180.
        rotation_axis = np.array([0, 0, 1])
        translation = np.array([3, 2, 1])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)
        sphere_centres = np.random.rand(10,3)
        sphere_centres[-1,:] = (100,100,100)
        sphere_radius = np.random.rand(10,)
        mask, _ = self.beam._get_candidate_spheres( sphere_centres, sphere_radius, motion)
        mask = np.sum(mask, axis=0)>0
        self.assertTrue( np.allclose(mask[0:9], True) )
        self.assertTrue( np.allclose(mask[9], False) )



if __name__ == '__main__':
    unittest.main()
