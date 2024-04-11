"""General internal package utility functions.

""" # TODO: Move some of these back into their classes where they are used.
import sys
import numpy as np
import logging
from numba import njit
from xfab import tools
from CifFile import ReadCif
from itertools import combinations
from scipy.spatial.transform import Rotation


def _diffractogram(
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

def _contained_by_intervals(value, intervals):
    """Assert if a float ``value`` is contained by any of a number of ``intervals``.
    """
    for bracket in intervals:
        if value >= bracket[0] and value <= bracket[1]:
            return True
    return False


def _cif_open(cif_file):
    """Helper function to be able to use the ``.CIFread`` of ``xfab``.
    """
    cif_dict = ReadCif(cif_file)
    return cif_dict[list(cif_dict.keys())[0]]


def _print_progress(progress_fraction, message):
    """Print a progress bar in the executing shell terminal.

    Args:
        progress_fraction (:obj:`float`): progress between 0 and 1.
        message (:obj:`str`): Optional message prepend the loading bar with. (max 55 characters)

    """
    assert len(
        message) <= 55., "Message to print is too long, max 55 characters allowed."
    progress_in_precent = np.round(100 * progress_fraction, 1)
    progress_bar_length = int(progress_fraction * 40)
    print("\r{0}{1} |{2}{3}|".format(message, " " *
                                                   (55 -
                                                    len(message)), "█" *
                                                   progress_bar_length, " " *
                                                   (40 -
                                                       progress_bar_length)) +
                     " " +
                     str(progress_in_precent) +
                     "%", end = '')
    if progress_fraction == 1.0:
        print("")


@njit
def _clip_line_with_convex_polyhedron(
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
        B0):
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
    crystal_strain = np.matmul(crystal_orientation.transpose(0,2,1), np.matmul(strain_tensor, crystal_orientation))
    lattice_matrix = _epsilon_to_b(crystal_strain,B0)
    return lattice_matrix

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

def _set_xfab_logging(disabled):
    """Disable/Enable all loging of xfab; it is very verbose!
    """
    xfab_modules = ['tools',
                    'structure',
                    'atomlib',
                    'detector',
                    'checks',
                    'sg',
                    'sglib',
                    'symmetry']
    for sub_module in xfab_modules:
        logging.getLogger('xfab.'+sub_module).disabled = disabled

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
    return np.asarray([ [e11, e12, e13],
                        [e12, e22, e23],
                        [e13, e23, e33] ], np.float64)

def _strain_as_vector(strain_tensor):
    return list(strain_tensor[0, :]) + list(strain_tensor[1, 1:]) + [strain_tensor[2, 2]]

def _b_to_epsilon(B_matrix, unit_cell):
    """Handle large deformations as opposed to current xfab.tools.b_to_epsilon
    """
    B = np.asarray(B_matrix, np.float64)
    B0 = tools.form_b_mat(unit_cell)
    F = np.dot(B0, np.linalg.inv(B))
    strain_tensor = 0.5*(F.T.dot(F) - np.eye(3))  # large deformations
    return _strain_as_vector(strain_tensor)

def _epsilon_to_b(crystal_strain, B0):
    """Handle large deformations as opposed to current xfab.tools.epsilon_to_b
    """
    C = 2*crystal_strain + np.eye(3, dtype=np.float32)
    eigen_vals = np.linalg.eigvalsh(C)
    if np.any( np.array(eigen_vals) < 0 ):
        raise ValueError("Unfeasible strain tensor with value: "+str(_strain_as_vector(crystal_strain))+ \
            ", will invert the unit cell with negative deformation gradient tensor determinant" )
    F = np.linalg.cholesky(C).transpose(0,2,1)
    B = np.linalg.inv(F).dot(B0)
    return B

def get_misorientations(orientations):
    """Compute the minimal angles neccessary to rotate a series of SO3 elements back into their mean orientation.

    Args:
        orientations (:obj: `numpy.array`): Orientation matrices, shape=(N,3,3)

    Returns:
        (:obj: `numpy.array`): misorientations in units of radians, shape=(N,)
    """
    mean_orientation = Rotation.mean(Rotation.from_matrix(orientations)).as_matrix()
    misorientations = np.zeros((orientations.shape[0],))
    for i, U in enumerate(orientations):
        difference_rotation = Rotation.from_matrix(U.dot(mean_orientation.T))
        misorientations[i] = Rotation.magnitude(difference_rotation)
    return misorientations


def compute_sides(points):
    """Computes the length of the sides of n tetrahedrons given a nx4x3 array"""
    # Reshape the points array to have shape (n, 1, 4, 3)
    reshaped_points = points[:, np.newaxis, :, :]

    # Compute the differences between each pair of points
    differences = reshaped_points - reshaped_points.transpose(0, 2, 1, 3)

    # Compute the squared distances along the last axis
    squared_distances = np.sum(differences**2, axis=-1)

    # Compute the distances by taking the square root of the squared distances
    dist_mat = np.sqrt(squared_distances)
    
    # Extract the 1-to-1 values from the distance matrix
    distances = np.hstack((dist_mat[:,0,1:],dist_mat[:,1,2:],dist_mat[:,2,3][:,np.newaxis]))
    
    return distances


def circumsphere_of_segments(segments):
    """Computes the minimum circumsphere of n segments given by a numpy array of vertices nx2x3"""
    centers = np.mean(segments,axis=1)
    radii = np.linalg.norm(centers-segments[:,0,:],axis=1)
    return centers, radii

def circumsphere_of_triangles(triangles):
    """Computes the minimum circumsphere of n triangles given by a numpy array of vertices nx3x3. Prints a message if any tetrahedron has 0 volume."""
    ab = triangles[:,1,:] - triangles[:,0,:]
    ac = triangles[:,2,:] - triangles[:,0,:]
    
    abXac = np.cross(ab,ac) 
    acXab = np.cross(ac,ab)

    norm_abXac = np.linalg.norm(abXac,axis=1)
    
    a_to_centre = (np.cross(abXac,ab)*((np.linalg.norm(ac,axis=1)**2)[:,np.newaxis])+np.cross(acXab,ac)*((np.linalg.norm(ab,axis=1)**2)[:,np.newaxis]))/(2*(np.linalg.norm(abXac,axis=1)**2)[:,np.newaxis])

    centers = triangles[:,0,:]+ a_to_centre    
    radii = np.linalg.norm(a_to_centre,axis=1)
    
    return centers, radii

def circumsphere_of_tetrahedrons(tetrahedra):
    """Computes the circumcenter of n tetrahedrons given by a numpy array of vertices nx4x3"""
    v0 = tetrahedra[:,0,:]
    v1 = tetrahedra[:,1,:]
    v2 = tetrahedra[:,2,:]
    v3 = tetrahedra[:,3,:]

    A = np.vstack(((v1-v0).T[np.newaxis,:],(v2-v0).T[np.newaxis,:],(v3-v0).T[np.newaxis,:])).transpose(2,0,1)
    B = 0.5*np.vstack((np.linalg.norm(v1,axis=1)**2-np.linalg.norm(v0,axis=1)**2,
                  np.linalg.norm(v2,axis=1)**2-np.linalg.norm(v0,axis=1)**2,
                  np.linalg.norm(v3,axis=1)**2-np.linalg.norm(v0,axis=1)**2)).T

    centers = np.matmul(np.linalg.inv(A),B[:,:,np.newaxis])[:,:,0]
    radii = np.linalg.norm((tetrahedra.transpose(2,0,1)-centers.transpose(1,0)[:,:,np.newaxis])[:,:,0],axis=0)
    
    return centers, radii


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def printvars(vars):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(vars.items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
