import numpy as np


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

    def __init__(
            self,
            convex_hull,
            scattered_wave_vector,
            incident_wave_vector,
            wavelength,
            incident_polarization_vector,
            rotation_axis,
            time,
            phase,
            hkl_indx):
        self.convex_hull = convex_hull
        self.scattered_wave_vector = scattered_wave_vector
        self.incident_wave_vector = incident_wave_vector
        self.wavelength = wavelength
        self.incident_polarization_vector = incident_polarization_vector
        self.rotation_axis = rotation_axis
        self.time = time
        self.phase = phase
        self.hkl_indx = hkl_indx

    @property
    def hkl(self):
        """hkl (:obj:`numpy array`): Miller indices [h,k,l] ``shape=(3,)``."""
        return self.phase.miller_indices[self.hkl_indx]

    @property
    def real_structure_factor(self):
        """hkl (:obj:`numpy array`): Real part of unit cell structure factor"""
        if self.phase.structure_factors is not None:
            return self.phase.structure_factors[self.hkl_indx, 0]
        else:
            return self.phase.structure_factors

    @property
    def imaginary_structure_factor(self):
        """hkl (:obj:`numpy array`): Imaginary part of unit cell structure factor"""
        if self.phase.structure_factors is not None:
            return self.phase.structure_factors[self.hkl_indx, 1]
        else:
            return self.phase.structure_factors

    @property
    def lorentz_factor(self):
        """Compute the Lorentz intensity factor for a scatterer.
        """
        k = self.incident_wave_vector
        kp = self.scattered_wave_vector
        theta = np.arccos(k.dot(kp) / (np.linalg.norm(k)**2)) / 2.
        korthogonal = kp - k * kp.dot(k) / np.linalg.norm(k**2)
        eta = np.arccos(
            self.rotation_axis.dot(korthogonal) /
            np.linalg.norm(korthogonal))
        if eta < 1e-8 or theta < 1e-8:
            return np.inf
        else:
            return 1. / (np.sin(2 * theta) * np.abs(np.sin(eta)))

    @property
    def polarization_factor(self):
        """Compute the Polarization intensity factor for a scatterer.
        """
        khatp = self.scattered_wave_vector / \
            np.linalg.norm(self.scattered_wave_vector)
        return 1 - np.dot(self.incident_polarization_vector, khatp)**2

    @property
    def centroid(self):
        """centroid (:obj:`numpy array`): centroid of the scattering region. ``shape=(3,)``
        """
        return np.mean(
            self.convex_hull.points[self.convex_hull.vertices], axis=0)

    @property
    def volume(self):
        """volume (:obj:`float`): volume of the scattering region volume
        """
        return self.convex_hull.volume
