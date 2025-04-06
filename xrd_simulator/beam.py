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
import torch
from xrd_simulator.cuda import device

torch.set_default_dtype(torch.float64)
from xrd_simulator.utils import ensure_torch, ensure_numpy


class Beam:
    """Represents a monochromatic xray beam as a convex polyhedra with uniform intensity.

    The beam is described in the laboratory coordinate system.

    Args:
        beam_vertices (:obj:`torch.Tensor`): Vertices of the xray beam in units of microns, ``shape=(N,3)``.
        xray_propagation_direction (:obj:`torch.Tensor`): Propagation direction of xrays, ``shape=(3,)``.
        wavelength (:obj:`float`): Xray wavelength in units of angstrom.
        polarization_vector (:obj:`torch.Tensor`): Beam linear polarization unit vector ``shape=(3,)``.
            Must be orthogonal to the xray propagation direction.

    Attributes:
        vertices (:obj:`torch.Tensor`): Vertices of the xray beam in units of microns, ``shape=(N,3)``.
        wavelength (:obj=`float`): Xray wavelength in units of angstrom.
        wave_vector (:obj=`torch.Tensor`): Beam wavevector with norm 2*pi/wavelength, ``shape=(3,)``
        polarization_vector (:obj=`torch.Tensor`): Beam linear polarization unit vector ``shape=(3,)``.
            Must be orthogonal to the xray propagation direction.
        centroid (:obj=`torch.Tensor`): Beam convex hull centroid ``shape=(3,)``
        halfspaces (:obj=`torch.Tensor`): Beam halfspace equation coefficients ``shape=(N,4)``.
            A point x is on the interior of the halfspace if: halfspaces[i,:-1].dot(x) +  halfspaces[i,-1] <= 0.

    """

    def __init__(
        self,
        beam_vertices: torch.Tensor | np.ndarray | list | tuple,
        xray_propagation_direction: torch.Tensor | np.ndarray | list | tuple,
        wavelength: float,
        polarization_vector: torch.Tensor | np.ndarray | list | tuple,
    ):
        self.wave_vector = ensure_torch(
            (2 * np.pi / wavelength)
            * xray_propagation_direction
            / np.linalg.norm(xray_propagation_direction)
        ).to(device)
        self.wavelength = wavelength
        self.set_beam_vertices(ensure_torch(beam_vertices))
        self.polarization_vector = ensure_torch(
            polarization_vector / np.linalg.norm(polarization_vector)
        ).to(device)
        assert torch.allclose(
            torch.dot(self.polarization_vector, self.wave_vector), ensure_torch(0.0)
        ), "The xray polarization vector is not orthogonal to the wavevector."

    def set_beam_vertices(self, beam_vertices: torch.Tensor):
        """Set the beam vertices defining the beam convex hull and update all dependent quantities.

        Args:
            beam_vertices (:obj:`torch.Tensor`): Vertices of the xray beam in units of microns, ``shape=(N,3)``.

        """
        ch = ConvexHull(beam_vertices.cpu().numpy())
        assert (
            ch.points.shape[0] == ch.vertices.shape[0]
        ), "The provided beam vertices does not form a convex hull"
        self.vertices = beam_vertices.clone().to(device)
        self.centroid = torch.mean(self.vertices, axis=0)
        self.halfspaces = ConvexHull(
            ensure_numpy(self.vertices), qhull_options="QJ"
        ).equations
        self.halfspaces = ensure_torch(
            np.unique(self.halfspaces.round(decimals=6), axis=0)
        ).to(device)

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Check if the beam contains a number of point(s).

        Args:
            points (:obj:`torch.Tensor`): Point(s) to evaluate ``shape=(3,n)`` or ``shape=(3,)``.

        Returns:
            torch.Tensor: Tensor with 1 if the point is contained by the beam and 0 otherwise, if single point is passed this returns
            scalar 1 or 0.

        """
        points = ensure_torch(points)
        normal_distances = torch.matmul(self.halfspaces[:, :3], points)
        if len(points.shape) == 1:
            return torch.all(normal_distances + self.halfspaces[:, 3] < 0)
        else:
            return (
                torch.sum(
                    (
                        normal_distances
                        + self.halfspaces[:, 3].reshape(self.halfspaces.shape[0], 1)
                    )
                    >= 0,
                    axis=0,
                )
                == 0
            )

    def intersect(self, vertices: torch.Tensor) -> ConvexHull | None:
        """Compute the beam intersection with a convex polyhedra.

        Args:
            vertices (:obj=`torch.Tensor`): Vertices of a convex polyhedra with ``shape=(N,3)``.

        Returns:
            A :class:`scipy.spatial.ConvexHull` object formed from the vertices of the intersection between beam vertices and
            input vertices, or None if no intersection exists.

        """
        vertices = ensure_torch(vertices)
        for vertex in vertices:
            if not self.contains(vertex):
                break
        else:
            return ConvexHull(
                vertices.cpu().numpy()
            )  # Tetra completely contained by beam

        poly_halfspace = ConvexHull(vertices.cpu().numpy()).equations
        poly_halfspace = np.unique(poly_halfspace.round(decimals=6), axis=0)
        combined_halfspaces = np.vstack((poly_halfspace, self.halfspaces.cpu().numpy()))

        # Since _find_feasible_point() is expensive it is worth checking for if the polyhedra
        # centroid is contained by the beam, being a potential cheaply computed interior_point.

        centroid = torch.mean(vertices, axis=0)
        if self.contains(centroid):
            interior_point = centroid
        else:
            trial_points = centroid + (vertices - centroid) * 0.99

            for i, is_contained in enumerate(self.contains(trial_points.T)):
                if is_contained:
                    interior_point = trial_points[i, :].flatten()
                    break
            else:
                interior_point = self._find_feasible_point(combined_halfspaces)

        if interior_point is not None:
            hs = HalfspaceIntersection(
                combined_halfspaces, interior_point.cpu().numpy()
            )
            return ConvexHull(hs.intersections)
        else:
            return None

    def save(self, path: str):
        """Save the xray beam to disc (via pickling).

        Args:
            path (:obj=`str`): File path at which to save, ending with the desired filename.

        """
        if not path.endswith(".beam"):
            path = path + ".beam"
        with open(path, "wb") as f:
            dill.dump(
                {
                    "wave_vector": ensure_numpy(self.wave_vector),
                    "vertices": ensure_numpy(self.vertices),
                    "polarization_vector": ensure_numpy(self.polarization_vector),
                    "halfspaces": ensure_numpy(self.halfspaces),
                    "wavelength": self.wavelength,
                },
                f,
                dill.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str) -> "Beam":
        """Load the xray beam from disc (via pickling).

        Args:
            path (:obj=`str`): File path at which to load, ending with the desired filename.

        Returns:
            Beam: Loaded Beam object.

        .. warning::
            This function will unpickle data from the provided path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        if not path.endswith(".beam"):
            raise ValueError("The loaded file must end with .beam")
        with open(path, "rb") as f:
            data = dill.load(f)
            loaded = cls(
                ensure_torch(data["vertices"]),
                ensure_torch(data["wave_vector"]),
                data["wavelength"],
                ensure_torch(data["polarization_vector"]),
            )
            loaded.halfspaces = ensure_torch(data["halfspaces"])
            return loaded

    def _find_feasible_point(self, halfspaces: np.ndarray) -> np.ndarray | None:
        """Find a point which is clearly inside a set of halfspaces (A * point + b < 0).

        Args:
            halfspaces (:obj=`numpy array`): Halfspace equations, each row holds coefficients of a halfspace (``shape=(N,4)``).

        Returns:
            (:obj=`None`) if no point is found else (:obj=`numpy array`) point.

        """
        halfspaces = ensure_numpy(halfspaces)
        res = linprog(
            c=np.zeros((3,)),
            A_ub=halfspaces[:, :-1],
            b_ub=-halfspaces[:, -1],
            bounds=(None, None),
            method="highs",
            options={
                "maxiter": 5000,
                "time_limit": 0.1,
            },  # typical solve time is ~1-2 ms
        )
        if res.success:
            if np.any(res.slack == 0):
                A = halfspaces[res.slack == 0, :-1]
                dx = np.linalg.solve(
                    A.T.dot(A), A.T.dot(-np.ones((A.shape[0],)) * 1e-5)
                )
                trial = res.x + dx
            else:
                trial = res.x
            if np.all((halfspaces[:, :-1].dot(trial) + halfspaces[:, -1]) < -1e-8):
                return trial
            else:
                return None
        else:
            return None

    def _get_candidate_spheres(
        self,
        sphere_centres: torch.Tensor,
        sphere_radius: torch.Tensor,
        rigid_body_motion,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask spheres which are close to intersecting a given convex beam hull undergoing a prescribed motion.

        NOTE: This function is approximate in the sense that the motion path is sampled at discrete moments in time
        at which the spheres are checked against the union of the sampled convex hulls. The sampling rate is chosen
        such that the motion translation maximally moves any sphere by half its radius. Furthermore, the sampling is
        selected to always be less than or equal to a rotation stepsize of 1 degree.

        Args:
            sphere_centres (:obj=`torch.Tensor`): Centroids of spheres ``shape=(3,n)``.
            sphere_radius (:obj=`torch.Tensor`): Radius of spheres ``shape=(n,)``.
            rigid_body_motion (:obj=`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].

        Returns:
            tuple: Mask with True if the sphere has a high intersection probability and the sample times.

        """
        inverse_rigid_body_motion = rigid_body_motion.inverse()

        dx = torch.min(sphere_radius) / 2.0
        translation = torch.abs(rigid_body_motion.translation / dx)
        number_of_sampling_points = int(
            torch.max(
                ensure_torch(
                    [
                        torch.max(translation),
                        torch.rad2deg(rigid_body_motion.rotation_angle),
                        ensure_torch(2.0),
                    ]
                )
            ).item()
            + 1
        )
        sample_times = torch.linspace(0, 1, number_of_sampling_points)

        R = sphere_radius.reshape(1, sphere_radius.shape[0])
        not_candidates = torch.zeros((len(sample_times), R.shape[1]), dtype=torch.bool)
        for i, time in enumerate(sample_times):

            halfspaces = ConvexHull(
                inverse_rigid_body_motion(self.vertices, time=time).cpu().numpy()
            ).equations
            halfspaces = np.unique(halfspaces.round(decimals=6), axis=0)
            normals = ensure_torch(halfspaces[:, 0:3])
            distances = ensure_torch(halfspaces[:, 3]).reshape(normals.shape[0], 1)
            sphere_beam_distances = normals.matmul(sphere_centres.T) + distances
            not_candidates[i, :] += torch.any(sphere_beam_distances > R, axis=0)

        return ~not_candidates, sample_times

    def _get_proximity_intervals(
        self,
        sphere_centres: torch.Tensor,
        sphere_radius: torch.Tensor,
        rigid_body_motion,
    ) -> list[list[list[float]]]:
        """Compute the parametric intervals t=[[t_1,t_2],[t_3,t_4],..] in which spheres are intersecting beam.

        This method can be used as a pre-checker before running the `intersect()` method on a polyhedral
        set. This avoids wasting compute resources on polyhedra which clearly do not intersect the beam.

        Args:
            sphere_centres (:obj=`torch.Tensor`): Centroids of spheres ``shape=(3,n)``.
            sphere_radius (:obj=`torch.Tensor`): Radius of spheres ``shape=(n,)``.
            rigid_body_motion (:obj=`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].

        Returns:
            list: Parametric ranges in which the spheres have an intersection with the beam. [(:obj=`None`)] if no intersection exists in ```time=[0,1]```.
            The entry at ```[i][j]``` is a list with two floats ```[t_1,t_2]``` and gives the j:th
            intersection interval of sphere number i with the beam.

        """
        candidate_mask, sample_times = self._get_candidate_spheres(
            sphere_centres, sphere_radius, rigid_body_motion
        )

        all_intersections = []
        for j in range(candidate_mask.shape[1]):
            if torch.sum(candidate_mask[:, j]) == 0:
                merged_intersection = [None]
                all_intersections.append(merged_intersection)
            else:
                counting = False
                merged_intersection = []

                i = 0
                if candidate_mask[i, j] and not counting:
                    merged_intersection.append([sample_times[i], sample_times[i + 1]])
                    counting = True

                for i in range(1, candidate_mask.shape[0] - 1):

                    if candidate_mask[i, j] and not counting:
                        merged_intersection.append(
                            [sample_times[i - 1], sample_times[i + 1]]
                        )
                        counting = True
                    elif not candidate_mask[i, j] and counting:
                        merged_intersection[-1][1] = sample_times[i + 1]
                        counting = False

                i = candidate_mask.shape[0] - 1
                if candidate_mask[i, j] and not counting:
                    merged_intersection.append([sample_times[i - 1], sample_times[i]])
                elif candidate_mask[i, j] and counting:
                    merged_intersection[-1][1] = sample_times[i]

                all_intersections.append(merged_intersection)

        return all_intersections
