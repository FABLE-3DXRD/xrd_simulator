"""General package internal utility functions.
"""
import sys

import numpy as np
from numba import njit
from xrd_simulator.xfab import tools
from CifFile import ReadCif
from itertools import combinations


def contained_by_intervals(value, intervals):
    """Assert if a float ``value`` is contained by any of a number of ``intervals``.
    """
    for bracket in intervals:
        if value >= bracket[0] and value <= bracket[1]:
            return True
    return False


def cif_open(cif_file):
    """Helper function to be able to use the ``.CIFread`` of ``xfab``.
    """
    cif_dict = ReadCif(cif_file)
    return cif_dict[list(cif_dict.keys())[0]]


def print_progress(progress_fraction, message):
    """Print a progress bar in the executing shell terminal.

    Args:
        progress_fraction (:obj:`float`): progress between 0 and 1.
        message (:obj:`str`): Optional message prepend the loading bar with. (max 55 characters)

    """
    assert len(
        message) <= 55., "Message to print is too long, max 55 characters allowed."
    progress_in_precent = np.round(100 * progress_fraction, 1)
    progress_bar_length = int(progress_fraction * 40)
    sys.stdout.write("\r{0}{1} | {2}>{3} |".format(message, " " *
                                                   (55 -
                                                    len(message)), "=" *
                                                   progress_bar_length, " " *
                                                   (40 -
                                                       progress_bar_length)) +
                     " " +
                     str(progress_in_precent) +
                     "%")
    if progress_fraction != 1.0:
        sys.stdout.flush()
    else:
        sys.stdout.write("\n")


@njit
def clip_line_with_convex_polyhedron(
        line_points,
        line_direction,
        plane_points,
        plane_normals):
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
        t_1 = np.sum(
            np.multiply(
                plane_points -
                line_point,
                plane_normals),
            axis=1)

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
    return np.array([np.cos(alpha_1),
                     sin_alpha_1 * sin_alpha_2 * np.cos(alpha_3),
                     sin_alpha_1 * sin_alpha_2 * np.sin(alpha_3),
                     sin_alpha_1 * np.cos(alpha_2)]).T


def lab_strain_to_B_matrix(
        strain_tensor,
        crystal_orientation,
        unit_cell):
    """Take a strain tensor in lab coordinates and produce the lattice matrix (B matrix).

    Args:
        strain_tensor (:obj:`numpy array`): Symmetric strain tensor in lab
            coordinates. ``shape=(3,3)``
        crystal_orientation (:obj:`numpy array`): Unitary crystal orientation matrix.
            ``crystal_orientation`` maps from crystal to lab coordinates. ``shape=(3,3)``
        unit_cell (:obj:`list` of :obj:`float`): Crystal unit cell representation of the form
            [a,b,c,alpha,beta,gamma], where alpha,beta and gamma are in units of degrees while
            a,b and c are in units of anstrom.

    Returns:
        (:obj:`numpy array`) B matrix, mapping from hkl Miller indices to realspace crystal
        coordinates, ``shape=(3,3)``.

    """
    crystal_strain = np.dot(
        crystal_orientation.T, np.dot(
            strain_tensor, crystal_orientation))
    lattice_matrix = tools.epsilon_to_b([crystal_strain[0, 0],
                                         crystal_strain[0, 1],
                                         crystal_strain[0, 2],
                                         crystal_strain[1, 1],
                                         crystal_strain[1, 2],
                                         crystal_strain[2, 2]],
                                        unit_cell)
    return lattice_matrix


def diffractogram(
        diffraction_pattern,
        det_centre_z,
        det_centre_y,
        binsize=1.0):
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
    histogram = np.zeros((len(bin_centres), ))
    for i in range(m):
        for j in range(n):
            radius = np.sqrt((i - det_centre_z)**2 + (j - det_centre_y)**2)
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
    A = 2*(subset_of_points[0]-subset_of_points[1:])
    pp = np.sum(subset_of_points*subset_of_points,axis=1)
    b = pp[0] - pp[1:]
    B = (subset_of_points[0]-subset_of_points[1:]).T
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
    radius, centroids = [],[]
    for k in range(2,5):
        for subset_of_points in combinations(points, k):
            centroids.append(_get_circumscribed_sphere_centroid(np.array(subset_of_points)))
            radius.append(np.max(np.linalg.norm(points - centroids[-1], axis=1)))
    index = np.argmin(radius)
    return centroids[index], radius[index]
