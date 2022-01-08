import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from xrd_simulator import laue
from scipy.optimize import root_scalar
from xrd_simulator._pickleable_object import PickleableObject


class Beam(PickleableObject):
    """Represents a monochromatic X-ray beam as a convex polyhedra.

    Args:
        beam_vertices (:obj:`numpy array`): Xray-beam vertices ``shape=(N,3)``.
        xray_propagation_direction (:obj:`numpy array`): Propagation direction of X-rays ``shape=(3,)``.
        wavelength (:obj:`float`): Xray wavelength in units of angstrom.
        polarization_vector (:obj:`numpy array`): Beam linear polarization unit vector ``shape=(3,)``.
            Must be orthogonal to the xray propagation direction.

    Attributes:
        vertices (:obj:`numpy array`): Xray-beam vertices ``shape=(N,3)``.
        wavelength (:obj:`float`): Xray wavelength in units of angstrom.
        wave_vector (:obj:`numpy array`): Beam wavevector ``shape=(3,)``
        centroid (:obj:`numpy array`): Beam centroid ``shape=(3,)``
        halfspaces (:obj:`numpy array`): Beam halfspace equation coefficients ``shape=(N,3)``.
            A point x is on the interior of the halfspace if: halfspaces[i,:-1].dot(x) +  halfspaces[i,-1] <= 0.
        polarization_vector (:obj:`numpy array`): Beam linear polarization unit vector ``shape=(3,)``.
            Must be orthogonal to the xray propagation direction.

    """

    def __init__(
            self,
            beam_vertices,
            xray_propagation_direction,
            wavelength,
            polarization_vector):
        self.wave_vector = (2 * np.pi / wavelength) * xray_propagation_direction / \
            np.linalg.norm(xray_propagation_direction)
        self.wavelength = wavelength
        self.set_beam_vertices(beam_vertices)
        self.polarization_vector = polarization_vector / \
            np.linalg.norm(polarization_vector)
        assert np.allclose(np.dot(self.polarization_vector, self.wave_vector),
                           0), "The x-ray polarisation vector is not orthogonal to the wavevector."

    def set_beam_vertices(self, beam_vertices):
        """Set the beam vertices defining the beam convex hull and update all dependent quantities.

        Args:
            beam_vertices (:obj:`numpy array`): Xray-beam vertices ``shape=(N,3)``.

        """
        # TODO: Add unit test for this function.
        ch = ConvexHull(beam_vertices)
        assert ch.points.shape[0] == ch.vertices.shape[0], "The provided beam vertices does not form a convex hull"
        self.vertices = beam_vertices.copy()
        self.centroid = np.mean(self.vertices, axis=0)
        self.halfspaces = ConvexHull(self.vertices).equations

    def find_feasible_point(self, halfspaces):
        """Find a point which is clearly inside a set of halfspaces (A * point + b < 0).

        Args:
            halfspaces (:obj:`numpy array`): Halfspace equations, each row holds coefficients of a halfspace (``shape=(N,4)``).

        Returns:
            (:obj:`None`) if no point is found else (:obj:`numpy array`) point.

        """
        # NOTE: from profiling: this method is slow, beware of using it
        # unnecessarily.
        norm_vector = np.reshape(np.linalg.norm(
            halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = - halfspaces[:, -1:]
        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=(
                None,
                None),
            method='highs-ipm',
            options={
                "maxiter": 2000})
        if res.success and res.x[-1] > 0:
            return res.x[:-1]
        else:
            return None

    def intersect(self, vertices):
        """Compute the beam intersection with a convex polyhedra, returns a list of HalfspaceIntersections.

        Args:
            vertices (:obj:`numpy array`): Vertices of a convex polyhedra with ``shape=(N,3)``.

        Returns:
            A scipy.spatial.ConvexHull object formed from the vertices of the intersection between beam vertices and
            input vertices.

        """

        vertices_contained_by_beam = [
            self.contains(vertex) for vertex in vertices]
        if np.all(vertices_contained_by_beam):
            # The beam completely contains the input convex polyhedra
            return ConvexHull(vertices)

        poly_halfspace = ConvexHull(vertices).equations
        combined_halfspaces = np.vstack((poly_halfspace, self.halfspaces))

        # Since find_feasible_point() is expensive it is worth checking for if the polyhedra
        # centroid is contained by the beam, being a potential cheaply computed
        # interior_point.
        centroid = np.mean(vertices, axis=0)
        if self.contains(centroid):
            interior_point = centroid
        else:
            interior_point = self.find_feasible_point(combined_halfspaces)

        if interior_point is not None:
            hs = HalfspaceIntersection(combined_halfspaces, interior_point)
            return ConvexHull(hs.intersections)
        else:
            return None

    def contains(self, point):
        """ Check if the beam contains a point.

        Args:
            point (:obj:`numpy array`): Point to evaluate ``shape=(3,)``.

        Returns:
            Boolean True if the beam contains the point.

        """
        return np.all(self.halfspaces[:, 0:3].dot(
            point) + self.halfspaces[:, 3] < 0)

    def get_proximity_intervals(
            self,
            sphere_centres,
            sphere_radius,
            rigid_body_motion):
        """Compute the parametric intervals t=[[t_1,t_2],[t_3,t_4],..] in which spheres are interesecting beam.

        This method can be used as a pre-checker before running the `intersect()` method on a polyhedral
        set. This avoids wasting compute resources on polyhedra which clearly do not intersect the beam.

        Args:
            sphere_centres (:obj:`numpy array`): Centroids of a spheres ``shape=(3,n)``.
            sphere_radius (:obj:`numpy array`): Radius of a spheres ``shape=(n,)``.
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].

        Returns:
            (:obj:`list` of :obj:`list` of :obj:`list`): Parametric ranges in which the spheres
            has an intersection with the beam. [(:obj:`None`)] if no intersection exist in ```time=[0,1]```.
            the entry at ```[i][j]``` is a list with two floats ```[t_1,t_2]``` and gives the j:th
            intersection interval of sphere number i with the beam.

        """

        beam_halfplane_normals = self.halfspaces[:, 0:3]
        beam_halfplane_offsets = self.halfspaces[:, 3]

        # Precomputable factors for root equations independent of sphere data.
        a0 = rigid_body_motion.rotator.K.dot(beam_halfplane_normals.T)
        a1 = rigid_body_motion.rotator.K2.dot(beam_halfplane_normals.T)
        a2 = -beam_halfplane_normals.dot(rigid_body_motion.translation)

        # Here we will store all intersections for all spheres.
        all_intersections = []

        for i in range(sphere_centres.shape[0]):

            # Here we will store all intersections for the current sphere.
            merged_intersections = [[0., 1.]]

            for p in range(self.halfspaces.shape[0]):

                # Intersections for the sphere attached to halfplane number p.
                new_intersections = []

                # Define the function we seek to have intervals in s fulfilling
                # : intersection_function(s) < 0
                q_0 = sphere_centres[i].dot(a0[:, p])
                q_1 = -sphere_centres[i].dot(a1[:, p])
                q_2 = a2[p]
                q_3 = sphere_centres[i].dot(
                    beam_halfplane_normals[p, :]) - q_1 - sphere_radius[i] + beam_halfplane_offsets[p]

                def intersection_function(time):
                    return q_0 * np.sin(time * rigid_body_motion.rotation_angle) + q_1 * np.cos(
                        time * rigid_body_motion.rotation_angle) + time * q_2 + q_3

                # Find roots of intersection_function(s) by first finding its
                # extremal points
                brackets = self._find_brackets_of_roots(
                    q_0, q_1, q_2, rigid_body_motion.rotation_angle, intersection_function)

                # Find roots numerically on the intervals
                roots = [
                    root_scalar(
                        intersection_function,
                        method='bisect',
                        bracket=bracket,
                        maxiter=50).root for bracket in brackets]
                roots.extend([0., 1.])
                interval_ends = np.sort(np.array(roots))
                fvals = intersection_function(
                    (interval_ends[0:-1] + interval_ends[1:]) / 2.)

                if np.sum(fvals > 0) == len(fvals):
                    # Always on the exterior of the beam halfplane
                    merged_intersections = [None]
                    break

                if np.sum(fvals < 0) == len(fvals):
                    new_intersections.append([0., 1.])
                    continue  # Always on the interior of the beam halfplane

                # Add in all intervals where intersection_function(s) < 0
                for k in range(len(interval_ends) - 1):
                    if fvals[k] < 0:
                        new_intersections.append(
                            [interval_ends[k], interval_ends[k + 1]])
                    else:
                        pass

                # Clean up the interval set, the sphere must intersect all
                # halfplanes for a shared interval in s.
                merged_intersections = self._merge_intersections(
                    merged_intersections, new_intersections)

                # The sphere will never intersect if no shared intervals are
                # found.
                if len(merged_intersections) == 0:
                    break

            all_intersections.append(merged_intersections)

        return all_intersections

    def _find_brackets_of_roots(self, q_0, q_1, q_2, rotation_angle, function):
        """Find all sub domains on time=[0,1] which are guaranteed to hold a root of function.
        """
        c_0 = rotation_angle * q_0
        c_1 = -rotation_angle * q_1
        c_2 = q_2
        time_1, time_2 = laue.find_solutions_to_tangens_half_angle_equation(
            c_0, c_1, c_2, rotation_angle)
        search_intervals = np.array(
            [s for s in [0, time_1, time_2, 1] if s is not None])
        f = function(search_intervals)
        indx = 0
        brackets = []
        for i in range(len(search_intervals) - 1):
            if np.sign(f[indx]) != np.sign(f[i + 1]):
                brackets.append(
                    [search_intervals[indx], search_intervals[i + 1]])
                indx = i + 1
        return brackets

    def _merge_intersections(self, merged_intersections, new_intersections):
        """Return intersection of merged_intersections and new_intersections

        NOTE: Assumes no overlaps inside the brackets of merged_intersections
        and new_intersections. I.e they are minimal descriptions of their
        respective point sets.
        """
        new_intervals = []
        for mi in merged_intersections:
            for ni in new_intersections:
                intersection = self._intersection(mi, ni)
                if intersection is not None:
                    new_intervals.append(intersection)
        return new_intervals

    def _intersection(self, bracket1, bracket2):
        """Find the intersection between two simple bracket intervals of form [start1,end1] and [start2,end2].
        """
        if (bracket2[0] <= bracket1[0] <= bracket2[1]) or (
                bracket2[0] <= bracket1[1] <= bracket2[1]):
            points = np.sort(
                [bracket2[0], bracket1[0], bracket2[1], bracket1[1]])
            return [points[1], points[2]]
        else:
            return None
