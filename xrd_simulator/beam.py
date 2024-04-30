"""The beam module is used to represent a beam of xrays. The idea is to create a :class:`xrd_simulator.beam.Beam` object and
pass it along to the :func:`xrd_simulator.polycrystal.Polycrystal.diffract` function to compute diffraction from the sample
for the specified xray beam geometry. Here is a minimal example of how to instantiate a beam object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_beam.py

Below follows a detailed description of the beam class attributes and functions.

"""
import numpy as np
import dill
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from xrd_simulator import utils, laue
from scipy.optimize import root_scalar

class Beam():
    """Represents a monochromatic xray beam as a convex polyhedra with uniform intensity.

    The beam is described in the laboratory coordinate system.

    Args:
        beam_vertices (:obj:`numpy array`): Vertices of the xray beam in units of microns, ``shape=(N,3)``.
        xray_propagation_direction (:obj:`numpy array`): Propagation direction of xrays, ``shape=(3,)``.
        wavelength (:obj:`float`): Xray wavelength in units of angstrom.
        polarization_vector (:obj:`numpy array`): Beam linear polarization unit vector ``shape=(3,)``.
            Must be orthogonal to the xray propagation direction.

    Attributes:
        vertices (:obj:`numpy array`): Vertices of the xray beam in units of microns, ``shape=(N,3)``.
        wavelength (:obj:`float`): Xray wavelength in units of angstrom.
        wave_vector (:obj:`numpy array`): Beam wavevector with norm 2*pi/wavelength, ``shape=(3,)``
        polarization_vector (:obj:`numpy array`): Beam linear polarization unit vector ``shape=(3,)``.
            Must be orthogonal to the xray propagation direction.
        centroid (:obj:`numpy array`): Beam convex hull centroid ``shape=(3,)``
        halfspaces (:obj:`numpy array`): Beam halfspace equation coefficients ``shape=(N,3)``.
            A point x is on the interior of the halfspace if: halfspaces[i,:-1].dot(x) +  halfspaces[i,-1] <= 0.

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
                           0), "The xray polarization vector is not orthogonal to the wavevector."

    def set_beam_vertices(self, beam_vertices):
        """Set the beam vertices defining the beam convex hull and update all dependent quantities.

        Args:
            beam_vertices (:obj:`numpy array`): Vertices of the xray beam in units of microns, ``shape=(N,3)``.

        """
        ch = ConvexHull(beam_vertices)
        assert ch.points.shape[0] == ch.vertices.shape[0], "The provided beam vertices does not form a convex hull"
        self.vertices = beam_vertices.copy()
        self.centroid = np.mean(self.vertices, axis=0)
        self.halfspaces = ConvexHull(self.vertices, qhull_options='QJ').equations

        # ConvexHull triangulates, this removes hull triangles positioned the same plane
        self.halfspaces = np.unique(self.halfspaces.round(decimals=6), axis=0)


    def contains(self, points):
        """ Check if the beam contains a number of point(s).

        Args:
            points (:obj:`numpy array`): Point(s) to evaluate ``shape=(3,n)`` or ``shape=(3,)``.

        Returns:
            numpy array with 1 if the point is contained by the beam and 0 otherwise, if single point is passed this returns
            scalar 1 or 0.

        """
        normal_distances = self.halfspaces[:, 0:3].dot(points)
        if len(points.shape) == 1:
            return np.all(normal_distances + self.halfspaces[:, 3] < 0)
        else:
            return np.sum( (normal_distances + self.halfspaces[:, 3].reshape(self.halfspaces.shape[0], 1)) >= 0, axis=0) == 0

    def intersect(self, vertices):
        """Compute the beam intersection with a convex polyhedra.

        Args:
            vertices (:obj:`numpy array`): Vertices of a convex polyhedra with ``shape=(N,3)``.

        Returns:
            A :class:`scipy.spatial.ConvexHull` object formed from the vertices of the intersection between beam vertices and
            input vertices.

        """

        for vertex in vertices:
            if not self.contains(vertex):
                break
        else:
            return ConvexHull(vertices) # Tetra completely contained by beam

        poly_halfspace = ConvexHull(vertices).equations
        poly_halfspace = np.unique(poly_halfspace.round(decimals=6), axis=0)
        combined_halfspaces = np.vstack((poly_halfspace, self.halfspaces))

        # Since _find_feasible_point() is expensive it is worth checking for if the polyhedra
        # centroid is contained by the beam, being a potential cheaply computed interior_point.

        centroid = np.mean(vertices, axis=0)
        if self.contains(centroid):
            interior_point = centroid
        else:
            trial_points = centroid + (vertices - centroid)*0.99

            for i, is_contained in enumerate(self.contains(trial_points.T)):
                if is_contained:
                    interior_point = trial_points[i,:].flatten()
                    break
            else:
                interior_point = self._find_feasible_point(combined_halfspaces)

        if interior_point is not None:
            hs = HalfspaceIntersection(combined_halfspaces, interior_point)
            return ConvexHull(hs.intersections)
        else:
            return None

    def save(self, path):
        """Save the xray beam to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.

        """
        if not path.endswith(".beam"):
            path = path + ".beam"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the xray beam from disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        if not path.endswith(".beam"):
            raise ValueError("The loaded file must end with .beam")
        with open(path, 'rb') as f:
            return dill.load(f)

    def _find_feasible_point(self, halfspaces):
        """Find a point which is clearly inside a set of halfspaces (A * point + b < 0).

        Args:
            halfspaces (:obj:`numpy array`): Halfspace equations, each row holds coefficients of a halfspace (``shape=(N,4)``).

        Returns:
            (:obj:`None`) if no point is found else (:obj:`numpy array`) point.

        """
        res = linprog(
            c       =  np.zeros((3,)),
            A_ub    =  halfspaces[:, :-1],
            b_ub    = -halfspaces[:, -1],
            bounds  = (None, None),
            method  = 'highs',
            options = {"maxiter": 5000, 'time_limit':0.1} # typical solve time is ~1-2 ms
            )
        if res.success:
            if np.any(res.slack==0):
                A     =  halfspaces[res.slack==0, :-1]
                dx    = np.linalg.solve(A.T.dot(A), A.T.dot( -np.ones((A.shape[0],))*1e-5 ) )
                trial = res.x + dx
            else:
                trial = res.x
            if np.all( (halfspaces[:, :-1].dot(trial) + halfspaces[:, -1]) < -1e-8 ):
                return trial
            else:
                return None
        else:
            return None

    def _get_candidate_spheres(
            self,
            sphere_centres,
            sphere_radius,
            rigid_body_motion):
        """Mask spheres which are close to intersecting a given convex beam hull undergoing a prescribed motion.

        NOTE: This function is approximate in the sense that the motion path is sampled at discrete moments in time
        at which the spheres are checked against the union of the sampled convex hulls. The sampling rate is choosen
        such that the motion translation maximally moves any sphere by half it's radius. Furthermore, the sampling is
        selected to always be less than or equal to a rotation stepsize of 1 degree.

        Args:
            sphere_centres (:obj:`numpy array`): Centroids of a spheres ``shape=(3,n)``.
            sphere_radius (:obj:`numpy array`): Radius of a spheres ``shape=(n,)``.
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].

        Returns:
            (:obj:`list` of :obj:`bool`): Mask with True if the sphere has a high intersection probability

        """

        inverse_rigid_body_motion = rigid_body_motion.inverse()

        dx = np.min(sphere_radius) / 2.
        translation = np.abs( rigid_body_motion.translation / dx )
        number_of_sampling_points = int( np.max( [np.max(translation), np.degrees(rigid_body_motion.rotation_angle), 2] ) + 1 )
        sample_times  = np.linspace(0, 1, number_of_sampling_points)

        R = sphere_radius.reshape(1, sphere_radius.shape[0])
        not_candidates = np.zeros((len(sample_times), R.shape[1]), dtype=bool)
        for i,time in enumerate(sample_times):

            halfspaces = ConvexHull( inverse_rigid_body_motion( self.vertices, time=time ) ).equations
            halfspaces = np.unique(halfspaces.round(decimals=6), axis=0)
            normals    = halfspaces[:, 0:3]
            distances  = halfspaces[:, 3].reshape(normals.shape[0],1)
            sphere_beam_distances  = normals.dot(sphere_centres.T) + distances
            not_candidates[i,:]   += np.any( sphere_beam_distances > R, axis=0)

        return ~not_candidates, sample_times

    def _get_proximity_intervals(
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
            # TODO: better unit tests.
            candidate_mask, sample_times  = self._get_candidate_spheres(sphere_centres,
                                                         sphere_radius,
                                                         rigid_body_motion)

            all_intersections = []
            for j in range(candidate_mask.shape[1]):
                if np.sum(candidate_mask[:,j])==0:
                    merged_intersection = [None]
                    all_intersections.append(merged_intersection)
                else:
                    counting = False
                    merged_intersection = []

                    i = 0
                    if candidate_mask[i,j] and not counting:
                        merged_intersection.append( [sample_times[i], sample_times[i+1]] )
                        counting = True

                    for i in range(1, candidate_mask.shape[0]-1):

                            if candidate_mask[i,j] and not counting:
                                merged_intersection.append( [sample_times[i-1], sample_times[i+1]] )
                                counting = True
                            elif not candidate_mask[i,j] and counting:
                                merged_intersection[-1][1] = sample_times[i+1]
                                counting = False

                    i = candidate_mask.shape[0] - 1
                    if candidate_mask[i,j] and not counting:
                        merged_intersection.append( [sample_times[i-1], sample_times[i]] )
                    elif candidate_mask[i,j] and counting:
                        merged_intersection[-1][1] = sample_times[i]

                    all_intersections.append(merged_intersection)

            return all_intersections