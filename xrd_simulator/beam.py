"""The beam class defines an X-ray as a 3d spatial (convex) object by 
taking a set of points and intepreting them as the convex hull of the beam
cross section. The beam is then extruded in a given direction towards infinity.""" 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
import utils

class Beam(object):

    def __init__(self, beam_vertices, wavelength, k1, k2 ):
        """

        Attributes:
            vertices (:obj:`numpy array`): Holds the vertices of the base, convex, polyhedral beam representation.
                The the beam is oriented in the xhat direction upon instantiation.

        """
        #TODO: perhaps the it would make more senese if the beam stored and returned the k vectors as a function of
        # s in [0,1].
        #TODO: assert the beam is convex.
        self.original_vertices = beam_vertices
        self.k1         = k1
        self.k2         = k2
        self.rotator    = utils.get_planar_rodriguez_rotator(k1, k2)
        self.wavelength = wavelength
        self.set_geometry(s=0)

    def set_geometry(self, s):
        """Align the beam into the new_propagation_direction by a performing rodriguez rotation.
        """
        for i in range( self.vertices.shape[0] ):
            self.vertices[i,:] = self.rotator(self.original_vertices[i,:], s)
        self.propagation_direction = new_propagation_direction
        self.halfspaces = self.ConvexHull( self.vertices ).equations
        self.k = self.rotator(self.k1, s)

        # NOTE: if we allow beams with arbitrary transformations then we ar ein trouble to solve the Laue equations
        # analytically. Numerical efforts are both slow and inaccurate it seems. Especially when k1 does not uniformly
        # go into k2. Image for instance k1 close to k1 s in [0,0.9] and k1 close to k2 in s=[0.9,1.0]. Then to solve
        # the problem numerically requires a very fine grid. (The problem is nonconvex in the general case). So it makes
        # sense to restrict the beam to do uniform planar rodriguez rotations in each intervall s=[0,1].

    def intersect( self, verticises ):
        """Compute the beam intersection with a series of convex polyhedra, returns a list of HalfspaceIntersections.
        """
        poly_halfspace = ConvexHull( verticises ).equation
        hs = HalfspaceIntersection( np.vstack( (poly_halfspace, self.halfspaces) ) , np.mean( verticises ) )
        return ConvexHull( hs.verticises )