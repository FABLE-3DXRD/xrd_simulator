import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.utils import source
from xrd_simulator import utils

class Detector(object):

    """Represents a rectangular X-ray scattering flat area detection device.

    The detector can collect scattering as abstract objects and map them to frame numbers.
    Using a render function these abstract representation can be rendered into pixelated frames.
    The detector is described by a 3 x 3 geometry matrix whic columns contain vectors that attach
    to three of the four corners of the detector.

    Args:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        geometry_matrix (:obj:`numpy array`): The detector geometry ```shape=(3,3)```.

    Attributes:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        frames (:obj:`list` of :obj:`list` of :obj:`scatterer.Scatterer`): Analytical diffraction patterns which
            may be rendered into pixelated frames. Each frame is a list of scattering objects
        detector_origin (:obj:`numpy array`): The detector origin at s=0, i.e ```geometry_matrix[:,0]```.
        geometry_matrix (:obj:`numpy array`): The detector geometry ```shape=(3,3)```.

    """

    def __init__(self, pixel_size, geometry_matrix ):
        self.geometry_matrix  = geometry_matrix
        self.detector_origin  = geometry_matrix[:,0]
        self.pixel_size       = pixel_size
        self.frames = []

    def render(self, frame_number):
        """Take a list of scatterers render to a pixelated pattern.

        Args:
            frame_number (:obj:`int`): Index of the frame in the :obj:`frames` list to be rendered.

        Returns:
            A pixelated frame as a (:obj:`numpy array`) with shape infered form the detector geometry and
            pixel size.

        NOTE: This function is meant to allow for overriding when specalised intensity models are to be tested.

        """
        frame = np.zeros( (int(self.zmax/self.pixel_size), int(self.ymax/self.pixel_size)) )
        for scatterer in self.frames[frame_number]:
            zd, yd = self.get_intersection( scatterer.kprime, scatterer.centroid )
            if self.contains(zd,yd):
                # TODO: add lorentz etc
                intensity = scatterer.volume * scatterer.lorentz_factor * scatterer.polarization_factor
                if scatterer.real_structure_factor is not None:
                    intensity = intensity * ( scatterer.real_structure_factor**2 + scatterer.imaginary_structure_factor**2 )
                frame[int(zd/self.pixel_size), int(yd/self.pixel_size)] += intensity
        return frame

    def get_intersection(self, ray_direction, source_point):
        """Get detector intersection in detector coordinates of singel a ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vector in direction of the X-ray propagation 
            source_point (:obj:`numpy array`): Origin of the ray.

        Returns:
            (:obj:`tuple`) zd, yd in detector plane coordinates.

        """
        s = (self.detector_origin - source_point).dot(self.normal) / ray_direction.dot(self.normal)
        intersection =  source_point + ray_direction*s
        zd = np.dot( intersection - self.detector_origin , self.zdhat)
        yd = np.dot( intersection - self.detector_origin , self.ydhat)
        return zd, yd

    def contains(self, zd, yd):
        """Determine if the detector coordinate zd,yd lies within the detector bounds.

        Args:
            zd (:obj:`float`): Detector z coordinate
            yd (:obj:`float`): Detector y coordinate

        Returns:
            (:obj:`boolean`) True if the zd,yd is within the detector bounds.

        """
        return zd>=0 and zd<=self.zmax and yd>=0 and yd<=self.ymax

    def get_wrapping_cone(self, k, source_point):
        """Compute the cone around a wavevector such that the cone wrapps the detector corners.

        Args:
            k (:obj:`numpy array`): Wavevector forming the central axis of cone.
            source_point (:obj:`numpy array`): Origin of the wavevector.

        Returns:
            (:obj:`float`) Cone opening angle divided by two (radians), corresponding to a maximum bragg angle after
                which scattering will systematically miss the detector.

        """
        fourth_corner_of_detector = np.expand_dims(self.geometry_matrix[:,2] + (self.geometry_matrix[:,1] - self.detector_origin[:]),axis=1)
        geom_mat = np.concatenate((self.geometry_matrix.copy(), fourth_corner_of_detector), axis=1)
        for i in range(4):
            geom_mat[:,i] -= source_point
        normalised_local_coord_geom_mat = geom_mat/np.linalg.norm(geom_mat, axis=0)
        cone_opening = np.arccos( np.dot(normalised_local_coord_geom_mat.T, k / np.linalg.norm(k) ) ) # These are two time Bragg angles        
        return np.max( cone_opening ) / 2.

