import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from xrd_simulator import utils, laue
from scipy.optimize import root_scalar

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

    def get_proximity_intervals(self, sphere_centres, sphere_radius):
        """Compute the parametric intervals s=[[s_1,s_2],[s_3,s_4],..] in which spheres are interesecting beam.

        This method can be used as a pre-checker before running the `intersect()` method on a polyhedral
        set. This avoids wasting compute resources on polyhedra which clearly do not intersect the beam.

        Args:
            sphere_centres (:obj:`numpy array`): Centroids of a spheres ```shape=(3,n)```.
            sphere_radius (:obj:`numpy array`): Radius of a spheres ```shape=(n,)```.

        Returns:
            (:obj:`list` of :obj:`list` of :obj:`list`): Parametric ranges in which the spheres
            has an intersection with the beam. [(:obj:`None`)] if no intersection exist in ```s=[0,1]```.
            the entry at ```[i][j]``` is a list with two floats ```[s_1,s_2]``` and gives the j:th 
            intersection interval of sphere number i with the beam.

        """ 

        beam_halfplane_normals =  self.original_halfspaces[:,0:3]
        beam_halfplane_offsets = -self.original_halfspaces[:,3]

        # Precomputable factors for root equations independent of sphere data.
        a0 = self.rotator.K.dot( beam_halfplane_normals.T )
        a1 = self.rotator.K2.dot( beam_halfplane_normals.T )
        a2 = -beam_halfplane_normals.dot( self.translation )

        all_intersections = [] # Here we will store all intersections for all spheres.

        for i in range(sphere_centres.shape[0]):

            merged_intersections = [[0., 1.]] # Here we will store all intersections for the current sphere.

            for p in range( self.original_halfspaces.shape[0] ):
                
                new_intersections = [] # Intersections for the sphere attached to halfplane number p.

                # Define the function we seek to have intervals in s fulfilling : intersection_function(s) < 0
                q_0  =  sphere_centres[i].dot( a0[:,p] )
                q_1  = -sphere_centres[i].dot( a1[:,p] )
                q_2  =  a2[p]
                q_3  =  sphere_centres[i].dot( beam_halfplane_normals[p,:] ) - q_1 - sphere_radius[i] - beam_halfplane_offsets[p]
                def intersection_function(s): 
                    return q_0 * np.sin( s*self.rotator.alpha ) + q_1 * np.cos(s*self.rotator.alpha) + s*q_2 + q_3

                # Find roots of intersection_function(s) by first finding its extreemal points
                brackets = self._find_brackets_of_roots(q_0, q_1, q_2, q_3, intersection_function)
                
                # Find roots numerically on the intervals 
                roots = [ root_scalar(intersection_function, method='bisect', bracket=bracket, maxiter=50).root for bracket in brackets ]
                roots.extend([0.,1.])
                interval_ends = np.sort( np.array( roots ) )
                fvals = intersection_function( (interval_ends[0:-1] + interval_ends[1:] ) /2.)

                if np.sum( fvals > 0 ) == len(fvals): 
                    merged_intersections = [None] # Always on the exterior of the beam halfplane
                    break

                if np.sum( fvals < 0 ) == len(fvals): 
                    new_intersections.append( [0.,1.] )
                    continue # Always on the interior of the beam halfplane
                
                # Add in all intervals where intersection_function(s) < 0
                for k in range(len(interval_ends)-1):
                    if fvals[k]<0:
                       new_intersections.append([interval_ends[k], interval_ends[k+1]])
                    else:
                        pass

                # Clean up the interval set, the sphere must intersect all halfplanes for a shared interval in s.
                merged_intersections = self._merge_intersections( merged_intersections, new_intersections )

                # The sphere will never intersect if no shared intervals are found.
                if len(merged_intersections)==0:
                    break

            all_intersections.append( merged_intersections )

        return all_intersections

    def _find_brackets_of_roots(self, q_0, q_1, q_2, q_3, function):
        """Find all sub domains on s=[0,1] which are guaranteed to hold a root of function.
        """
        c_0 = self.rotator.alpha*q_0
        c_1 = -self.rotator.alpha*q_1
        c_2 = q_2
        s_1, s_2 = laue.find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, self.rotator.alpha )
        search_intervals = np.array( [s for s in [0, s_1, s_2, 1] if s is not None] )
        f = function( search_intervals )
        indx = 0
        brackets = []
        for i in range(len(search_intervals)-1):
            if np.sign( f[indx] )!=np.sign( f[i+1] ):
                brackets.append( [search_intervals[indx],search_intervals[i+1]] )
                indx = i+1
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
                    new_intervals.append( intersection )
        return new_intervals

    def _intersection(self, bracket1, bracket2):
        """Find the intersection between two simple bracket intervals of form [start1,end1] and [start2,end2].
        """
        if (bracket2[0] <= bracket1[0] <= bracket2[1]) or (bracket2[0] <= bracket1[1] <= bracket2[1]):
            points = np.sort([ bracket2[0],bracket1[0],bracket2[1],bracket1[1] ])
            return [points[1],points[2]]
        else:
            return None


            
