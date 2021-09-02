"""The beam class defines an X-ray as a 3d spatial (convex) object by 
taking a set of points and intepreting them as the convex hull of the beam
cross section. The beam is then extruded in a given direction towards infinity.""" 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
import utils

class Beam(object):

    def __init__(self, beam_vertices, wavelength ):
        """

        Attributes:
            vertices (:obj:`numpy array`): Holds the vertices of the base, convex, polyhedral beam representation.
                The the beam is oriented in the xhat direction upon instantiation.

        """
        self.vertices   = beam_vertices
        self.wavelength = wavelength
        self.propagation_direction = np.array([1.,0.,0.])
        self.halfspaces = self._update_halfspaces()

    def align(self, new_propagation_direction):
        """Align the beam into the new_propagation_direction by a performing rodriguez rotation.
        """
        rotator = utils.get_planar_rodriguez_rotator(self.propagation_direction, new_propagation_direction)
        for i in range( self.vertices.shape[0] ):
            self.vertices[i,:] = rotator(self.vertices[i,:], s=1)
        self.propagation_direction = new_propagation_direction
        self.halfspaces = self._update_halfspaces()

    def _update_halfspaces(self):
        """Given the current convex polyhedra representation of the beam update the beam halfplanes.
        """
        self.halfspaces = self.ConvexHull( self.vertices ).equations

    def intersect( self, verticises ):
        """Compute the beam intersection with a series of convex polyhedra, returns a list of HalfspaceIntersections.
        """
        poly_halfspace = ConvexHull( verticises ).equation
        return HalfspaceIntersection( np.vstack( (poly_halfspace, self.halfspaces) ) , np.mean( verticises ) )