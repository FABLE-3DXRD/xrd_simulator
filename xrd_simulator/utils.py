"""
General internal package utility functions.

This module provides various utility functions used for internal package operations.
The functions include mathematical computations, geometric transformations, file handling,
and progress tracking.

Functions:
    _diffractogram: Compute diffractogram from pixelated diffraction pattern.
    _contained_by_intervals: Check if a value is contained within specified intervals.
    _cif_open: Open a CIF file using the ReadCif function from the CifFile module.
    _print_progress: Print a progress bar in the executing shell terminal.
    _clip_line_with_convex_polyhedron: Compute lengths of parallel lines clipped by a convex polyhedron.
    alpha_to_quarternion: Generate a unit quaternion from spherical angle coordinates on the S3 ball.
    lab_strain_to_B_matrix: Convert strain tensors in lab coordinates to lattice matrices (B matrices).
    _get_circumscribed_sphere_centroid: Compute the centroid of a circumscribed sphere for a given set of points.
    _get_bounding_ball: Compute a minimal bounding ball for a set of Euclidean points.
    _set_xfab_logging: Enable or disable logging for the xfab module.
    _verbose_manager: Manage global verbose options for logging within with statements.
    _strain_as_tensor: Convert a strain vector to a strain tensor.
    _strain_as_vector: Convert a strain tensor to a strain vector.
    _b_to_epsilon: Compute strain tensor from B matrix for large deformations.
    _epsilon_to_b: Compute B matrix from strain tensor for large deformations.
    _get_misorientations: Compute minimal angles required to rotate SO3 elements to their mean orientation.
    _compute_sides: Compute the lengths of the sides of tetrahedrons.
    _circumsphere_of_segments: Compute the minimum circumsphere of line segments.
    _circumsphere_of_triangles: Compute the minimum circumsphere of triangles.
    _circumsphere_of_tetrahedrons: Compute the circumcenter of tetrahedrons.
"""

import logging
from itertools import combinations
from CifFile import ReadCif
from scipy.spatial.transform import Rotation
import numpy as np
import sys
import xrd_simulator.cuda
import torch

torch.set_default_dtype(torch.float64)


def _diffractogram(diffraction_pattern, det_centre_z, det_centre_y, binsize=1.0):
    """Compute diffractogram from pixelated diffraction pattern.

    Args:
        diffraction_pattern (:obj:`numpy array`): Pixelated diffraction pattern``shape=(m,n)``
        det_centre_z (:obj:`numpy array`): Intersection pixel coordinate between
                 beam centroid line and detector along z-axis.
        det_centre_y (:obj:`list` of :obj:`float`): Intersection pixel coordinate between
                 beam centroid line and detector along y-axis.
        binsize  (:obj:`list` of :obj:`float`): Histogram binsize. (Detector pixels are integrated
            radially around the azimuth)

    Returns:
        (:obj:`tuple`) with ``bin_centres`` and ``histogram``.

    """
    m, n = diffraction_pattern.shape
    max_radius = np.max([m, n])
    bin_centres = np.arange(0, int(max_radius + 1), binsize)
    histogram = np.zeros((len(bin_centres),))
    for i in range(m):
        for j in range(n):
            radius = np.sqrt((i - det_centre_z) ** 2 + (j - det_centre_y) ** 2)
            bin_index = np.argmin(np.abs(bin_centres - radius))
            histogram[bin_index] += diffraction_pattern[i, j]
    clip_index = len(histogram) - 1
    for k in range(len(histogram) - 1, 0, -1):
        if histogram[k] != 0:
            break
        else:
            clip_index = k
    clip_index = np.min([clip_index + m // 10, len(histogram) - 1])
    return bin_centres[0:clip_index], histogram[0:clip_index]


def _contained_by_intervals(value, intervals):
    """Assert if a float ``value`` is contained by any of a number of ``intervals``."""
    for bracket in intervals:
        if value >= bracket[0] and value <= bracket[1]:
            return True
    return False


def _cif_open(cif_file):
    """Helper function to be able to use the ``.CIFread`` of ``xfab``."""
    cif_dict = ReadCif(cif_file)
    return cif_dict[list(cif_dict.keys())[0]]


def _print_progress(progress_fraction, message):
    """Print a progress bar in the executing shell terminal.

    Args:
        progress_fraction (:obj:`float`): progress between 0 and 1.
        message (:obj:`str`): Optional message prepend the loading bar with. (max 55 characters)

    """
    assert (
        len(message) <= 55.0
    ), "Message to print is too long, max 55 characters allowed."
    progress_in_precent = np.round(100 * progress_fraction, 1)
    progress_bar_length = int(progress_fraction * 40)
    print(
        "\r{0}{1} |{2}{3}|".format(
            message,
            " " * (55 - len(message)),
            "█" * progress_bar_length,
            " " * (40 - progress_bar_length),
        )
        + " "
        + str(progress_in_precent)
        + "%",
        end="",
    )
    if progress_fraction == 1.0:
        print("")


def _clip_line_with_convex_polyhedron(
    line_points, line_direction, plane_points, plane_normals
):
    """Compute lengths of parallel lines clipped by a convex polyhedron defined by 2d planes.

    For algorithm description see: Mike Cyrus and Jay Beck. “Generalized two- and three-
    dimensional clipping”. (1978) The algorithms is based on solving orthogonal equations and
    sorting the resulting plane line interestion points to find which are entry and which are
    exit points through the convex polyhedron.

    Args:
        line_points (:obj:`numpy array`): base points of rays
            (exterior to polyhedron), ``shape=(n,3)``
        line_direction  (:obj:`numpy array`): normalized ray direction
            (all rays have the same direction),  ``shape=(3,)``
        plane_points (:obj:`numpy array`): point in each polyhedron face plane, ``shape=(m,3)``
        plane_normals (:obj:`numpy array`): outwards element face normals. ``shape=(m,3)``

    Returns:
        clip_lengths (:obj:`numpy array`) : intersection lengths.  ``shape=(n,)``

    """
    clip_lengths = np.zeros((line_points.shape[0],))
    t_2 = np.dot(plane_normals, line_direction)
    te_mask = t_2 < 0
    tl_mask = t_2 > 0
    for i, line_point in enumerate(line_points):

        # find parametric line-plane intersection based on orthogonal equations
        t_1 = np.sum(np.multiply(plane_points - line_point, plane_normals), axis=1)

        # Zero division for a ray parallel to plane, numpy gives np.inf so it
        # is ok!
        t_i = t_1 / t_2

        # Sort intersections points as potential entry and exit points
        t_e = np.max(t_i[te_mask])
        t_l = np.min(t_i[tl_mask])

        if t_l > t_e:
            clip_lengths[i] = t_l - t_e

    return clip_lengths


def alpha_to_quarternion(alpha_1, alpha_2, alpha_3):
    """Generate a unit quarternion by providing spherical angle coordinates on the S3 ball.

    Args:
        alpha_1,alpha_2,alpha_3 (:obj:`numpy array` or :obj:`float`): Radians. ``shape=(N,4)``

    Returns:
        (:obj:`numpy array`) Rotation as unit quarternion ``shape=(N,4)``

    """
    sin_alpha_1, sin_alpha_2 = np.sin(alpha_1), np.sin(alpha_2)
    return np.array(
        [
            np.cos(alpha_1),
            sin_alpha_1 * sin_alpha_2 * np.cos(alpha_3),
            sin_alpha_1 * sin_alpha_2 * np.sin(alpha_3),
            sin_alpha_1 * np.cos(alpha_2),
        ]
    ).T


def lab_strain_to_B_matrix(
    strain_tensor: torch.Tensor, crystal_orientation: torch.Tensor, B0: torch.Tensor
) -> torch.Tensor:
    """Take n strain tensors in lab coordinates and produce the lattice matrix (B matrix).

    Args:
        strain_tensor (:obj:`torch.Tensor`): Symmetric strain tensor in lab
            coordinates. ``shape=(n,3,3)``
        crystal_orientation (:obj:`torch.Tensor`): Unitary crystal orientation matrix.
            ``crystal_orientation`` maps from crystal to lab coordinates. ``shape=(n,3,3)``
        B0 matrix (:obj:`torch.Tensor`): Matrix containing the reciprocal underformed lattice parameters.``shape=(3,3)``

    Returns:
        (:obj:`torch.Tensor`) B matrix mapping from hkl Miller indices to realspace crystal
        coordinates. ``shape=(n,3,3)``

    """
    if strain_tensor.ndim == 2:
        strain_tensor = strain_tensor.unsqueeze(0)
    if crystal_orientation.ndim == 2:
        crystal_orientation = crystal_orientation.unsqueeze(0)

    crystal_strain = torch.matmul(
        crystal_orientation.transpose(1, 2),
        torch.matmul(strain_tensor, crystal_orientation),
    )
    B = _epsilon_to_b(crystal_strain, B0)
    return B.squeeze()


def _get_circumscribed_sphere_centroid(subset_of_points):
    """Compute circumscribed_sphere_centroid by solving linear systems of equations
    enforcing the centorid to be a linear combination of the subset_of_points space.

    The central idea is to substitute the squared radius in the nonlinear quadratic
    sphere equations and proceed to find a linear system with only the sphere centroid
    as the unknown.

    Args:
        subset_of_points (:obj:`numpy array`): Points to circumscribe with a sphere ``shape=(n,3)``

    Returns:
        (:obj:`numpy array`) with ``centroid`` of``shape=(3,)``

    """
    A = 2 * (subset_of_points[0] - subset_of_points[1:])
    pp = np.sum(subset_of_points * subset_of_points, axis=1)
    b = pp[0] - pp[1:]
    B = (subset_of_points[0] - subset_of_points[1:]).T
    x = np.linalg.solve(A.dot(B), b - A.dot(subset_of_points[0]))
    return subset_of_points[0] + B.dot(x)


def _get_bounding_ball(points):
    """Compute a minimal bounding ball for a set of euclidean points.

    This is a naive implementation that checks all possible minimal bounding balls
    by constructing spheres that have increasingly higher number of points from the
    point set exactly at their surface. For extremely low dimensional problems
    (such as tetrahedrons in a mesh) this has been found to be more efficient than
    more standard methods, such as the Emo Welzl 1991 algorithm. For larger point sets
    however, the method quickly becomes slow.

    Args:
        points (:obj:`numpy array`): Points to be wrapped by sphere ``shape=(n,3)``

    Returns:
        (:obj:`tuple` of :obj:`numpy array` and :obj:`float`) ``centroid`` and ``radius``.

    """
    radius, centroids = [], []
    for k in range(2, 5):
        for subset_of_points in combinations(points, k):
            centroids.append(
                _get_circumscribed_sphere_centroid(np.array(subset_of_points))
            )
            radius.append(np.max(np.linalg.norm(points - centroids[-1], axis=1)))
    index = np.argmin(radius)
    return centroids[index], radius[index]


def _set_xfab_logging(disabled):
    """Disable/Enable all loging of xfab; it is very verbose!"""
    xfab_modules = [
        "tools",
        "structure",
        "atomlib",
        "detector",
        "checks",
        "sg",
        "sglib",
        "symmetry",
    ]
    for sub_module in xfab_modules:
        logging.getLogger("xfab." + sub_module).disabled = disabled


class _verbose_manager(object):
    """Manage global verbose options in with statements; to turn of
    external package loggings easily inside xrd_simulator.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            _set_xfab_logging(disabled=False)

    def __exit__(self, type, value, traceback):
        if self.verbose:
            _set_xfab_logging(disabled=True)


def _strain_as_tensor(strain_vector):
    e11, e12, e13, e22, e23, e33 = strain_vector
    return np.asarray([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]], np.float64)


def _strain_as_vector(strain_tensor):
    return (
        list(strain_tensor[0, :]) + list(strain_tensor[1, 1:]) + [strain_tensor[2, 2]]
    )


def _b_to_epsilon(B_matrix, B0):
    """Handle large deformations as opposed to current xfab.tools.b_to_epsilon"""
    B = np.asarray(B_matrix, np.float64)
    F = np.dot(B0, np.linalg.inv(B))
    strain_tensor = 0.5 * (F.T.dot(F) - np.eye(3))  # large deformations
    return _strain_as_vector(strain_tensor)


def _epsilon_to_b(crystal_strain, B0):
    """Handle large deformations as opposed to current xfab.tools.epsilon_to_b"""
    crystal_strain = ensure_torch(crystal_strain, dtype=torch.float64)
    B0 = ensure_torch(B0, dtype=torch.float64)

    C = 2 * crystal_strain + torch.eye(3, dtype=torch.float64)

    eigen_vals = torch.linalg.eigvalsh(C)
    if torch.any(eigen_vals < 0):
        raise ValueError(
            "Unfeasible strain tensor with value: "
            + str(_strain_as_vector(crystal_strain))
            + ", will invert the unit cell with negative deformation gradient tensor determinant"
        )
    if C.ndim == 3:
        F = torch.transpose(torch.linalg.cholesky(C), 2, 1)
    else:
        F = torch.transpose(torch.linalg.cholesky(C), 1, 0)

    B = torch.matmul(torch.linalg.inv(F), B0)
    B = B.cpu()
    return B


def _get_misorientations(orientations):
    """
    Compute the minimal angles necessary to rotate a series of SO3 elements back into their mean orientation.

    Args:
        orientations (:obj: `numpy.array`): Orientation matrices, shape=(N,3,3)

    Returns:
        :obj: `numpy.array`: misorientations in units of radians, shape=(N,)
    """
    mean_orientation = Rotation.mean(Rotation.from_matrix(orientations)).as_matrix()
    misorientations = np.zeros((orientations.shape[0],))
    for i, U in enumerate(orientations):
        difference_rotation = Rotation.from_matrix(U.dot(mean_orientation.T))
        misorientations[i] = Rotation.magnitude(difference_rotation)
    return misorientations


def _compute_sides(points):
    """
    Computes the lengths of the sides of multiple tetrahedrons.

    Args:
        points (:obj: `numpy.array`): An array of shape (n, 4, 3), where `n` is the number of tetrahedrons.
                                Each tetrahedron is defined by 4 vertices in 3D space.

    Returns:
        :obj: `numpy.array`: An array of shape (n, 6) containing the lengths of the sides of the tetrahedrons.
                       Each row corresponds to a tetrahedron and contains the lengths of its 6 sides.
    """
    # Reshape the points array to have shape (n, 1, 4, 3)
    reshaped_points = points[:, np.newaxis, :, :]

    # Compute the differences between each pair of points
    differences = reshaped_points - reshaped_points.transpose(0, 2, 1, 3)

    # Compute the squared distances along the last axis
    squared_distances = np.sum(differences**2, axis=-1)

    # Compute the distances by taking the square root of the squared distances
    dist_mat = np.sqrt(squared_distances)

    # Extract the 1-to-1 values from the distance matrix
    distances = np.hstack(
        (dist_mat[:, 0, 1:], dist_mat[:, 1, 2:], dist_mat[:, 2, 3][:, np.newaxis])
    )

    return distances


def _circumsphere_of_segments(segments):
    """
    Computes the circumcenters and circumradii of multiple line segments.

    Args:
        segments (:obj: `numpy.array`): An array of shape (n, 2, 3), where `n` is the number of line segments.
                                   Each line segment is defined by 2 vertices in 3D space.

    Returns:
        tuple(:obj: `numpy.array`, :obj: `numpy.array`): A tuple containing:
             - centers (:obj: `numpy.array`): An array of shape (n, 3) containing the circumcenters of the line segments.
             - radii (:obj: `numpy.array`): An array of shape (n,) containing the circumradii of the line segments.
    """
    centers = np.mean(segments, axis=1)
    radii = np.linalg.norm(centers - segments[:, 0, :], axis=1)
    return centers, radii * 1.0001  # because loss of floating point precision


def _circumsphere_of_triangles(triangles):
    """
    Computes the circumcenters and circumradii of multiple triangles.

    Args:
        triangles (:obj: `numpy.array`): An array of shape (n, 3, 3), where `n` is the number of triangles.
                                   Each triangle is defined by 3 vertices in 3D space.

    Returns:
        tuple(:obj: `numpy.array`, :obj: `numpy.array`): A tuple containing:
             - centers (:obj: `numpy.array`): An array of shape (n, 3) containing the circumcenters of the triangles.
             - radii (:obj: `numpy.array`): An array of shape (n,) containing the circumradii of the triangles.
    """
    ab = triangles[:, 1, :] - triangles[:, 0, :]
    ac = triangles[:, 2, :] - triangles[:, 0, :]

    abXac = np.cross(ab, ac)
    acXab = np.cross(ac, ab)

    a_to_centre = (
        np.cross(abXac, ab) * ((np.linalg.norm(ac, axis=1) ** 2)[:, np.newaxis])
        + np.cross(acXab, ac) * ((np.linalg.norm(ab, axis=1) ** 2)[:, np.newaxis])
    ) / (2 * (np.linalg.norm(abXac, axis=1) ** 2)[:, np.newaxis])

    centers = triangles[:, 0, :] + a_to_centre
    radii = np.linalg.norm(a_to_centre, axis=1)

    return centers, radii * 1.0001  # because loss of floating point precision


def _circumsphere_of_tetrahedrons(tetrahedra):
    """
    Computes the circumcenters and circumradii of multiple tetrahedrons.

    Args:
        tetrahedra (:obj: `numpy.array`): An array of shape (n, 4, 3), where `n` is the number of tetrahedrons.
                                    Each tetrahedron is defined by 4 vertices in 3D space.

    Returns:
        tuple(:obj: `numpy.array`, :obj: `numpy.array`): A tuple containing:
             - centers (:obj: `numpy.array`): An array of shape (n, 3) containing the circumcenters of the tetrahedrons.
             - radii (:obj: `numpy.array`): An array of shape (n,) containing the circumradii of the tetrahedrons.
    """

    v0 = tetrahedra[:, 0, :]
    v1 = tetrahedra[:, 1, :]
    v2 = tetrahedra[:, 2, :]
    v3 = tetrahedra[:, 3, :]

    A = np.vstack(
        (
            (v1 - v0).T[np.newaxis, :],
            (v2 - v0).T[np.newaxis, :],
            (v3 - v0).T[np.newaxis, :],
        )
    ).transpose(2, 0, 1)
    B = (
        0.5
        * np.vstack(
            (
                np.linalg.norm(v1, axis=1) ** 2 - np.linalg.norm(v0, axis=1) ** 2,
                np.linalg.norm(v2, axis=1) ** 2 - np.linalg.norm(v0, axis=1) ** 2,
                np.linalg.norm(v3, axis=1) ** 2 - np.linalg.norm(v0, axis=1) ** 2,
            )
        ).T
    )

    centers = np.matmul(np.linalg.inv(A), B[:, :, np.newaxis])[:, :, 0]
    radii = np.linalg.norm(
        (tetrahedra.transpose(2, 0, 1) - centers.transpose(1, 0)[:, :, np.newaxis])[
            :, :, 0
        ],
        axis=0,
    )

    return centers, radii * 1.0001  # because loss of floating point precision


def sizeof_fmt(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def list_vars(vars):
    """
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(vars.items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))"""
    print("===================================================")
    # Get CPU variable sizes
    var_sizes = sorted(
        ((name, sys.getsizeof(value)) for name, value in vars.items()),
        key=lambda x: -x[1],
    )
    for name, size in var_sizes[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    # Get GPU variable sizes
    gpu_var_sizes = []
    for name, value in vars.items():
        if torch.is_tensor(value):
            gpu_var_sizes.append((name, value.nbytes))

    gpu_var_sizes = sorted(gpu_var_sizes, key=lambda x: -x[1])
    for name, size in gpu_var_sizes[:10]:
        print("{:>30} (GPU): {:>8}".format(name, sizeof_fmt(size)))
    print("===================================================")


def ensure_torch(data: np.ndarray | torch.Tensor | list | tuple) -> torch.Tensor:
    """Convert input to torch tensor if it isn't already.

    Args:
        data: Input data to convert. Can be:
            - numpy array
            - torch tensor
            - list
            - tuple

    Returns:
        torch.Tensor: The input data converted to a torch tensor

    Examples:
        >>> ensure_torch([1, 2, 3])
        tensor([1, 2, 3])
        >>> ensure_torch(np.array([1, 2, 3]))
        tensor([1, 2, 3])
        >>> ensure_torch(ensure_torch([1, 2, 3]))
        tensor([1, 2, 3])
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif torch.is_tensor(data):
        return data
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    return torch.tensor(data)


from typing import Union


def ensure_numpy(data: np.ndarray | torch.Tensor | list | tuple) -> np.ndarray:
    """Convert input to numpy array if it isn't already.

    Args:
        data: Input data to convert. Can be:
            - numpy array
            - torch tensor
            - list
            - tuple

    Returns:
        np.ndarray: The input data converted to a numpy array

    Examples:
        >>> ensure_numpy([1, 2, 3])
        array([1, 2, 3])
        >>> ensure_numpy(np.array([1, 2, 3]))
        array([1, 2, 3])
        >>> ensure_numpy(ensure_torch([1, 2, 3]))
        array([1, 2, 3])
    """
    if torch.is_tensor(data):
        return data.cpu().detach().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    return np.array(data)
