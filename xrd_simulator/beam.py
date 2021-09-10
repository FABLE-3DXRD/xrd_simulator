import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
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
    of the cross product of :math:`\\boldsymbol{k}_1` and :math:`\\boldsymbol{k}_2`.

    Args:
        beam_vertices (:obj:`numpy array`): Xray-beam vertices for s=0.
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.
        k1 (:obj:`numpy array`): Beam wavevector for s=0 with ```shape=(3,)```
        k2 (:obj:`numpy array`): Beam wavevector for s=1 with ```shape=(3,)```

    Attributes:
        original_vertices (:obj:`numpy array`): Xray-beam vertices for s=0.
        wavelength (:obj:`float`): 
        k1 (:obj:`numpy array`): Beam wavevector for s=0 with ```shape=(3,)```
        k2 (:obj:`numpy array`): Beam wavevector for s=1 with ```shape=(3,)```
        rotator (:obj:`utils.PlanarRodriguezRotator`): Callable object performing rodriguez 
            rotations from k1 towards k2.

    """

    def __init__(self, beam_vertices, wavelength, k1, k2 ):
        #TODO: assert the beam is convex.
        self.original_vertices = beam_vertices.copy()
        self.vertices   = beam_vertices.copy()
        self.k1         = k1
        self.k2         = k2
        self.rotator    = utils.PlanarRodriguezRotator(k1, k2)
        self.wavelength = wavelength
        self.set_geometry(s=0)

    def set_geometry(self, s):
        """Align the beam into the new_propagation_direction by a performing rodriguez rotation.

        Args:
            s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
                while s=1 to a beam with wavevector k2. The beam vertices are rotated by a rodriguez rotation 
                parametrised by s.

        """
        for i in range( self.vertices.shape[0] ):
            self.vertices[i,:] = self.rotator(self.original_vertices[i,:], s)
        self.halfspaces = ConvexHull( self.vertices ).equations
        self.k = self.rotator(self.k1, s)

        # NOTE: if we allow beams with arbitrary transformations then we ar ein trouble to solve the Laue equations
        # analytically. Numerical efforts are both slow and inaccurate it seems. Especially when k1 does not uniformly
        # go into k2. Image for instance k1 close to k1 s in [0,0.9] and k1 close to k2 in s=[0.9,1.0]. Then to solve
        # the problem numerically requires a very fine grid. (The problem is nonconvex in the general case). So it makes
        # sense to restrict the beam to do uniform planar rodriguez rotations in each intervall s=[0,1].

    def intersect( self, vertices ):
        """Compute the beam intersection with a series of convex polyhedra, returns a list of HalfspaceIntersections.

        Args:
            vertices (:obj:`numpy array`): Vertices of a convex polyhedra with ```shape=(N,3)```.
        
        Returns:
            A scipy.spatial.ConvexHull object formed from the vertices of the intersection between beam vertices and
            input vertices.

        """
        poly_halfspace = ConvexHull( vertices ).equations
        hs = HalfspaceIntersection( np.vstack( (poly_halfspace, self.halfspaces) ) , np.mean( vertices, axis=0 ) )
        return ConvexHull( hs.intersections )