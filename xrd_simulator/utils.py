"""General package internal utility functions.
"""
import os
import sys

import numpy as np
from numba import njit
from xrd_simulator.xfab import tools

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
    from CifFile import ReadCif
    cif_dict = ReadCif(cif_file)
    return cif_dict[list(cif_dict.keys())[0]]


def print_progress(progress_fraction, message):
    """Print a progress bar in the executing shell terminal.

    Args:
        progress_fraction (:obj:`float`): progress between 0 and 1.
        message (:obj:`str`): Optional message prepend the loading bar with. (max 55 characters)

    """
    assert len(message) <= 55., "Message to print is too long, max 55 characters allowed."
    progress_in_precent = np.round(100 * progress_fraction, 1)
    progress_bar_length = int(progress_fraction * 40)
    sys.stdout.write("\r{0}{1} | {2}>{3} |".format(message, " " * (55 - len(message)), "=" *
                                                   progress_bar_length, " " *
                                                   (40 - progress_bar_length)) + " " +
                     str(progress_in_precent) + "%")
    if progress_fraction != 1.0:
        sys.stdout.flush()
    else:
        sys.stdout.write("\n")


@njit
def clip_line_with_convex_polyhedron(line_points, line_direction, plane_points, plane_normals):
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

        # Zero division for a ray parallel to plane, numpy gives np.inf so it is ok!
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


def lab_strain_to_lattice_matrix(strain_tensor, crystal_orientation, unit_cell):
    """Take a strain tensor in crystal coordinates and produce the lattice matrix (B matrix).

    Args:
        strain_tensor (:obj:`numpy array`): Symmetric strain tensor in crystal
            coordinates. ``shape=(3,3)``
        crystal_orientation (:obj:`numpy array`): Unitary crystal orientation matrix.
            ``crystal_orientation`` maps from crystal to lab coordinates. ``shape=(3,3)``
        unit_cell (:obj:`list` of :obj:`float`): Crystal unit cell representation of the form
            [a,b,c,alpha,beta,gamma], where alpha,beta and gamma are in units of degrees while
            a,b and c are in units of anstrom.

    Returns:
        (:obj:`numpy array`) Lattice matrix (B matrix)``shape=(3,3)``

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


def diffractogram( diffraction_pattern, det_centre_z, det_centre_y, binsize=1.0 ):
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
    m,n = diffraction_pattern.shape
    max_radius = np.max([m,n])
    bin_centres = np.arange( 0, int(max_radius+1), binsize )
    histogram = np.zeros( (len(bin_centres), ) )
    for i in range(m):
        for j in range(n):
            radius = np.sqrt( (i-det_centre_z)**2 + (j-det_centre_y)**2 )
            bin_index = np.argmin( np.abs( bin_centres - radius ) )
            histogram[bin_index] += diffraction_pattern[i,j]
    clip_index=len(histogram)-1
    for k in range(len(histogram)-1, 0, -1):
        if histogram[k]!=0: break
        else: clip_index = k
    clip_index = np.min([clip_index+m//10, len(histogram)-1])
    return bin_centres[0:clip_index], histogram[0:clip_index]

def _get_circumsphere(point_cloud):
    """
    Computes the circumsphere of a set of points
    Parameters
    ----------
    point_cloud : (M, N) ndarray, where 1 <= M <= N + 1
            The input points
    Returns
    -------
    C, r2 : ((2) ndarray, float)
            The center and the squared radius of the circumsphere

    NOTE: This code is modified from https://github.com/marmakoide/miniball
    """

    U = point_cloud[1:] - point_cloud[0]
    B = np.sqrt(np.sum(U ** 2, axis=1))
    U /= B[:, None]
    B /= 2
    C = np.dot(np.linalg.solve(np.inner(U, U), B), U)
    r2 = np.sum(C ** 2)
    C += point_cloud[0]
    return C, r2


def get_bounding_ball(point_cloud, epsilon=1e-7, rng=np.random.default_rng()):
    """
    Computes the smallest bounding ball of a set of points
    Parameters
    ----------
    point_cloud : (M, N) ndarray, where 1 <= M <= N + 1
            The input points
    epsilon : float
            Tolerance used when testing if a set of point belongs to the same sphere.
            Default is 1e-7
    rng : np.random.Generator
        Pseudo-random number generator used internally. Default is the one default
        one provided by np.
    Returns
    -------
    C, r2 : ((2) ndarray, float)
            The center and the squared radius of the circumsphere

    NOTE: This code is modified from https://github.com/marmakoide/miniball
    """

    # Iterative implementation of Welzl's algorithm, see
    # "Smallest enclosing disks (balls and ellipsoids)" Emo Welzl 1991

    def circle_contains(D, p):
        c, r2 = D
        return np.sum((p - c) ** 2) <= r2

    def get_boundary(R):
        if len(R) == 0:
            return np.zeros(point_cloud.shape[1]), 0.0

        if len(R) <= point_cloud.shape[1] + 1:
            return _get_circumsphere(point_cloud[R])

        c, r2 = _get_circumsphere(point_cloud[R[: point_cloud.shape[1] + 1]])
        if np.all(np.fabs(np.sum((point_cloud[R] - c) ** 2, axis=1) - r2) < epsilon):
            return c, r2

    class Node(object):
        def __init__(self, P, R):
            self.P = P
            self.R = R
            self.D = None
            self.pivot = None
            self.left = None
            self.right = None

    def traverse(node):
        stack = [node]
        while len(stack) > 0:
            node = stack.pop()

            if len(node.P) == 0 or len(node.R) >= point_cloud.shape[1] + 1:
                node.D = get_boundary(node.R)
            elif node.left is None:
                node.pivot = rng.choice(node.P)
                node.left = Node(list(set(node.P) - set([node.pivot])), node.R)
                stack.extend((node, node.left))
            elif node.right is None:
                if circle_contains(node.left.D, point_cloud[node.pivot]):
                    node.D = node.left.D
                else:
                    node.right = Node(node.left.P, node.R + [node.pivot])
                    stack.extend((node, node.right))
            else:
                node.D = node.right.D
                node.left, node.right = None, None

    point_cloud = point_cloud.astype(float, copy=False)
    root = Node(range(point_cloud.shape[0]), [])
    traverse(root)
    return root.D

def get_bounding_ball_2(points):
    c = np.mean(points, axis=1)
    r = np.max( np.linalg.norm( points - c, axis=1 ) )
    return c, r