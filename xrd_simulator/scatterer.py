import numpy as np 
import matplotlib.pyplot as plt

class Scatterer(object):

    """Defines a scattering single crystal as a convex polyhedra.

    Args:
        convex_hull (:obj:`scipy.spatial.ConvexHull`): Object describing the convex hull of the self.
        kprime (:obj:`string`): Scattering vector, i.e the wavevector pointing in the direction of diffraction.
        s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
            while s=1 to a beam with wavevector k2. The s value of the scatterer implies what detector
            position is applicable during diffraction.
        phase (:obj:`Phase`): The Phase object representing the material of the self.
        hkl_indx (:obj:`int`): Index of Miller index in the `phase.miller_indices` list.

    Attributes:
        convex_hull (:obj:`scipy.spatial.ConvexHull`): Object describing the convex hull of the self.
        kprime (:obj:`string`): Scattering vector, i.e the wavevector pointing in the direction of diffraction.
        s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
            while s=1 to a beam with wavevector k2. The s value of the scatterer implies what detector
            position is applicable during diffraction.

    """ 

    def __init__(self, convex_hull, kprime, bragg_angle, s, phase, hkl_indx ):
        self.convex_hull = convex_hull
        self.kprime = kprime
        self.bragg_angle = bragg_angle
        self.s = s

        self.phase = phase
        self.hkl_indx = hkl_indx

    @property
    def hkl(self):
        """hkl (:obj:`numpy array`): Miller indices [h,k,l] ```shape=(3,)```."""
        return self.phase.miller_indices[ self.hkl_indx ]

    @property
    def real_structure_factor(self):
        """hkl (:obj:`numpy array`): Real part of unit cell structure factor"""
        if self.phase.structure_factors:
            return self.phase.structure_factors[ self.hkl_indx, 0 ]
        else:
            return self.phase.structure_factors

    @property
    def imaginary_structure_factor(self):
        """hkl (:obj:`numpy array`): Imaginary part of unit cell structure factor"""
        if self.phase.structure_factors:
            return self.phase.structure_factors[ self.hkl_indx, 1 ]
        else:
            return self.phase.structure_factors

    @property
    def lorentz_factor( self, scatterer ):
        """Compute the Lorentz intensity factor for a scatterer.
        """
        theta       = np.arccos( self.k.dot( self.kprime ) / (np.linalg.norm(self.k)**2) ) / 2.
        korthogonal = self.kprime - self.k * self.kprime.dot( self.k ) / np.linalg.norm(self.k**2)
        eta         = np.arccos( self.rhat.dot( korthogonal ) / np.linalg.norm( korthogonal ) )
        return 1. / ( np.sin(2 * theta) * np.abs( np.sin(eta) ) )

    @property
    def polarization_factor( self, scatterer ):
        """Compute the Polarization intensity factor for a scatterer.
        """
        return np.abs( self.incident_polarization_vector.dot( self.scattered_polarization_vector ) )**2

    @property
    def incident_polarization_vector(self):
        pass
    
    @property
    def scattered_polarization_vector(self):
        pass

    @property
    def centroid(self):
        """centroid (:obj:`numpy array`): centroid of the scattering region. ```shape=(3,)```
        """
        return np.mean( self.convex_hull.points[self.convex_hull.vertices], axis=0 )

    @property
    def volume(self):
        """volume (:obj:`float`): volume of the scattering region volume 
        """
        return self.convex_hull.volume