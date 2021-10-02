import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from xrd_simulator import utils

class Beam(object):
    """Represents a monochromatic X-ray beam as a convex polyhedra.

    The Beam object stores a state of an X-ray beam. In a parametric scan intervall
    the beam is allowed to take on wavevectors in the fan formed 
    by [:math:`\\boldsymbol{k}_1`, :math:`\\boldsymbol{k}_2`] such that all wavevectors
    in the scan intervall lies within the plane defined by :math:`\\boldsymbol{k}_1` and 
    unto :math:`\\boldsymbol{k}_2`. The geometry or profile of the beam is likewise restricted
    to rotate by the same transformation that brings :math:`\\boldsymbol{k}_1`  unto
    :math:`\\boldsymbol{k}_2` (rodriguez rotation). I.e all vertices of the convex beam hull will 
    rotate according to a rodriguez rotation defined by the unit vector which is in the direction
    of the cross product of :math:`\\boldsymbol{k}_1` and :math:`\\boldsymbol{k}_2`. Before rotation
    is executed, and optional linear translation may be performed.

    Args:
        beam_vertices (:obj:`numpy array`): Xray-beam vertices for s=0.
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.
        k1 (:obj:`numpy array`): Beam wavevector for s=0 with ```shape=(3,)```
        k2 (:obj:`numpy array`): Beam wavevector for s=1 with ```shape=(3,)```
        translation (:obj:`numpy array`): Beam linear translation on s=[0,1], ```shape=(3,)```.
            The beam moves s*translation before each rotation.

    Attributes:
        original_vertices (:obj:`numpy array`): Xray-beam vertices for s=0.
        wavelength (:obj:`float`): 
        k1 (:obj:`numpy array`): Beam wavevector for s=0 with ```shape=(3,)```
        k2 (:obj:`numpy array`): Beam wavevector for s=1 with ```shape=(3,)```
        rotator (:obj:`utils.RodriguezRotator`): Callable object performing rodriguez 
            rotations from k1 towards k2.
        centroid (:obj:`numpy array`): Beam centroid ```shape=(3,)```
        translation (:obj:`numpy array`): Beam linear translation on s=[0,1], ```shape=(3,)```.
            The beam moves s*translation before each rotation.
    """

    def __init__(self, beam_vertices, wavelength, k1, k2, translation ):

        assert np.allclose( np.linalg.norm(k1), 2 * np.pi / wavelength ), "Wavevector k1 is not of length 2*pi/wavelength."
        assert np.allclose( np.linalg.norm(k2), 2 * np.pi / wavelength ), "Wavevector k1 is not of length 2*pi/wavelength."
        ch = ConvexHull( beam_vertices )
        assert ch.points.shape[0]==ch.vertices.shape[0], "The provided beam veertices does not form a convex hull"

        self.original_vertices   = beam_vertices.copy()
        self.original_centroid   = np.mean(self.original_vertices, axis=0)
        self.original_halfspaces = ConvexHull( self.original_vertices ).equations
        self.halfspaces = self.original_halfspaces.copy()
        self.vertices   = self.original_vertices.copy()
        self.centroid   = self.original_centroid.copy()
        self.k1         = k1
        self.k2         = k2
        self.rotator    = utils.RodriguezRotator(k1, k2)
        self.wavelength = wavelength
        self.translation = translation

        self.set_geometry(s=0)

    def set_geometry(self, s):
        """Align the beam into the new_propagation_direction by a performing rodriguez rotation.

        Args:
            s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
                while s=1 to a beam with wavevector k2. The beam vertices are rotated by a rodriguez rotation 
                parametrised by s.

        """
        self.vertices = self.rotator( (self.original_vertices + self.translation*s).T, s).T
        self.halfspaces = ConvexHull( self.vertices ).equations
        self.k = self.rotator(self.k1, s)
        self.centroid = self.rotator(self.original_centroid, s)

    def find_feasible_point(self, halfspaces):
        """Find a point which is clearly inside a set of halfspaces (A * point + b < 0).

        Args:
            halfspaces (:obj:`numpy array`): Halfspace equations, each row holds coefficents of a halfspace (```shape=(N,4)```).

        Returns:
            (:obj:`None`) if no point is found else (:obj:`numpy array`) point.

        """
        #NOTE: from profiling: this method is the current bottleneck with about 50 % of total CPU time is spent here.
        norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = - halfspaces[:, -1:]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='highs-ipm')
        if res.success and res.x[-1]>0:
            return res.x[:-1]
        else:
            return None

    def intersect( self, vertices ):
        """Compute the beam intersection with a series of convex polyhedra, returns a list of HalfspaceIntersections.

        Args:
            vertices (:obj:`numpy array`): Vertices of a convex polyhedra with ```shape=(N,3)```.

        Returns:
            A scipy.spatial.ConvexHull object formed from the vertices of the intersection between beam vertices and
            input vertices.

        """
        poly_halfspace = ConvexHull( vertices ).equations
        combined_halfspaces = np.vstack( (poly_halfspace, self.halfspaces) )
        interior_point = self.find_feasible_point(combined_halfspaces)
        if interior_point is not None:
            hs = HalfspaceIntersection( combined_halfspaces , interior_point )
            return ConvexHull( hs.intersections )
        else:
            return None

    def get_proximity_intervals(self, sphere_centre, sphere_radius):
        """Compute the parametric intervals s=[s_1,s_2,_3,..] in which sphere could is interesecting the beam.

        This method can be used as a pre-checker before running the `intersect()` method on a polyhedral
        set. This avoids wasting compute resources on polyhedra which clearly do not intersect the beam.

        Args:
            sphere_centre (:obj:`numpy array`): Centroid of a sphere ```shape=(3,)```.
            sphere_radius (:obj:`numpy array`): Radius of a sphere ```shape=(3,)```.

        Returns:
            (:obj:`tuple` of :obj:`float`): s_1, s_2, i.e the parametric range in which the sphere
            has an intersection with the beam. (:obj:`None`) if no intersection exist in s=[0,1]

        """ 
        max_rotation_increment    = np.linalg.norm( sphere_centre - self.rotator(sphere_centre, s=1) )
        max_translation_increment = np.linalg.norm( sphere_radius )
        
        # TODO: Think on this.
        # number_of_interval_points = np.ceil( np.max( [max_rotation_increment + max_translation_increment, 10] ) ).astype(int)
        number_of_interval_points = 10
        
        evaluation_points = np.linspace( 0, 1, number_of_interval_points )

        intersection_map = np.ones( ( number_of_interval_points, ) )
        for p, beam_halfplane in enumerate(self.original_halfspaces):
                for n,s in enumerate(evaluation_points):
                    if intersection_map[n]==1:
                        normal = beam_halfplane[0:3]
                        d0     = -beam_halfplane[3]
                        term1  = sphere_centre.dot( self.rotator(normal, s) )
                        term2  = normal.dot( self.translation )*s
                        term3  = sphere_radius + d0
                        intersection_map[n] *= (term1 - term2 - term3 <= 0)
                if np.sum(intersection_map)==0:
                    return None

        indx = np.where( intersection_map==1 )[0]
        
        intervals = []
        previ = 0
        for i in indx:

            if i-previ==1:
                # forward update
                if i==len(evaluation_points)-1:
                    pass
                else:
                    intervals[-1][1] = evaluation_points[i+1]
            else:
                # forward & backward add
                if i==0:
                    intervals.append( [evaluation_points[i], evaluation_points[i+1]] )
                elif i==len(evaluation_points)-1:
                    intervals.append( [evaluation_points[i-1], evaluation_points[i]] )
                else:
                    intervals.append( [evaluation_points[i-1], evaluation_points[i+1]] )

            previ = i

        return np.array(intervals)


    

