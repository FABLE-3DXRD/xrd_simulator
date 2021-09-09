"""The detector class takes a set of scattere-rays objects and produces a 
discrete pixelated framestack."""

import numpy as np 
import matplotlib.pyplot as plt
from xrd_simulator import utils

class Detector(object):

    def __init__(self, pixel_size, geometry_matrix ):
        self.pixel_size        = pixel_size
        self.geometry_matrix   = geometry_matrix
        self.frames = []
        self.set_geometry(s=0)

    def set_geometry(self, s):
        """Set the geomtry of the detector based on the parametric value s in [0,1]
        """
        G = self.geometry_matrix(s)
        self.normalised_geometry_matrix = G / np.linalg.norm(G, axis=0)
        self.zdhat, self.zmax  = utils.get_unit_vector_and_l2norm(G[:,2], G[:,0])
        self.ydhat, self.ymax  = utils.get_unit_vector_and_l2norm(G[:,1], G[:,0])
        self.normal = np.cross(self.zdhat, self.ydhat)

    def render(self, frame_number):
        """Take a list of scatterers render and add to the frames list.
        """
        # TODO: make the renderer a bit more andvanced not just scaling intensity against 
        # scattering volume.
        frame = np.zeros( (int(self.zmax/self.pixel_size), int(self.ymax/self.pixel_size)) )
        for scatterer in frames[frame_number]:
            self.set_geometry( s = scatterer.s )
            zd, yd = self.get_intersection( scatterer.kprime, scatterer.get_centroid() )
            if self.contains(zd,yd):
                intensity = scatterer.get_volume()
                frame[int(zd/self.pixel_size), int(yd/self.pixel_size)] += intensity
        self.set_geometry( s = 0 )
        return frame

    def get_intersection(self, ray_direction, source_point):
        """Compute intersection in detector coordinates between ray originating from point c and propagating in along v.
        """
        s = ( self.zdhat.dot(self.normal) - source_point.dot(self.normal) ) / ray_direction.dot(self.normal)
        det_intersection =  source_point + ray_direction*s - self.geometry_matrix[0,:]
        zd = np.dot(det_intersection, self.zdhat)
        yd = np.dot(det_intersection, self.ydhat)
        return zd, yd

    def contains(self, zd, yd):
        """Determine if the detector cooridnate zd,yd lies within the detector bounds.
        """
        return zd>=0 and zd<=self.zmax and yd>=0 and yd<=self.zmax

    def get_wrapping_cone(self, k):
        """Compute the cone around a fixed wavevector with opening such that the cone intersects at least one detector corner.
        
        Given a range of illumination direction [k1, k2] which lies on the cone of the detector, i.e we assume here
        that k1 and k2 will terminate in the detector plane. Altough diffraction is possible for other settings with
        and of beam path detector it has not been implemented here.
        """
        #TODO: Verify that the min cone openings occur at k1 or k2 
        zd, yd = self.get_intersection(k, c=0)
        assert self.contains(zd, yd), "You provided a wavevector, k="+str(k)+", that will not intersect the detector."
        cone_opening = np.arccos( np.dot(self.normalised_geometry_matrix.T, k / np.linalg.norm(k) ) ) # These are two time Bragg angles
        return np.min( cone_opening ) / 2. 

    def approximate_wrapping_cone(self, beam, samples=180, margin=np.pi/180):
        """Given a moving detector as well as a variable wavevector approximate an upper Bragg angle bound.

        NOTE: for specalized use you may override this function and provide some fixed Bragg angle bound
        """
        cone_angles = []
        for s in np.linspace(0, 1, samples):
            self.set_geometry(s)
            beam.set_geometry(s)
            cone_angles.append( self.get_wrapping_cone( beam.k ) )
        self.set_geometry(s=0)
        beam.set_geometry(s=0)
        return np.min(allk) + margin

