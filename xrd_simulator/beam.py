"""The beam class defines an X-ray as a 3d spatial (convex) object by 
taking a set of points and intepreting them as the convex hull of the beam
cross section. The beam is then extruded in a given direction towards infinity.""" 

import numpy as np 
import matplotlib.pyplot as plt
import utils

class Beam(object):

    def __init__(self, beam_profile_vertices, wavelength ):
        """

        Attributes:
            beam_base_vertices (:obj:`numpy array`): Holds the vertices of the base, convex, polyhedral beam representation.
                The baseline is that the beam is oriented in the xhat direction.

        """
        nverts = beam_profile_vertices.shape[0]
        self.beam_base_vertices = np.zeros((2*nverts, 3))
        self.beam_base_vertices[0:nverts,1:2] = beam_profile_vertices
        self.beam_base_vertices[0:nverts,0]   = np.array([1,0,0])*np.inf
        self.beam_base_vertices[nverts:,1:2]  = beam_profile_vertices
        self.beam_base_vertices[nverts:,0]    = -np.array([1,0,0])*np.inf 
        self.wavelength = wavelength

    def get_polyhedral_beam(self, k):
        """Rotate the beam_base_polyhedron into the k direction and 
        return a new beam polyhedron.
        """
        alpha = np.acos(  v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) )
        c,s = np.cos( alpha ), np.sin( alpha )
        rhat  = np.cross(k, np.array([1,0,0]))
        rhat /= np.linalg.norm(r)
        poly_beam = np.zeros( self.beam_base_vertices.shape )
        for i in range( poly_beam.shape[0] ):
            poly_beam[i,:] = self.beam_base_vertices[i,:]*c + np.cross( r, self.beam_base_vertices[i,:] )*s
        return poly_beam