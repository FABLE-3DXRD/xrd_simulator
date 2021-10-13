import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.utils import source
from xrd_simulator import utils

class Detector(object):

    """Represents a rectangular X-ray scattering flat area detection device.

    The detector can collect scattering as abstract objects and map them to frame numbers.
    Using a render function these abstract representation can be rendered into pixelated frames.
    The detector is described by a 3 x 3 geometry matrix whic columns contain vectors that attach
    to three of the four corners of the detector. By providing a callable geometry matrix function
    the detector position at any point in the scan intervall [:math:`\\boldsymbol{k}_1, \\boldsymbol{k}_2`],
    :math:`s \\in[0, 1]` may be retrieved. This means that the detector does not need to be fixed in relation 
    to a global laboratory coordinate system.

    Args:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        geometry_descriptor (:obj:`callable`): geometry_descriptor(s) <- G where G is a ```shape=(3,3)``` geometry
            matrix which columns attach to three corners of the detector array (units of microns).

    Attributes:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        geometry_descriptor (:obj:`callable`): geometry_descriptor(s) <- G where G is a ```shape=(3,3)``` geometry
            matrix which columns attach to three corners of the detector array (units of microns).
        frames (:obj:`list` of :obj:`list` of :obj:`scatterer.Scatterer`): Analytical diffraction patterns which
            may be rendered into pixelated frames. Each frame is a list of scattering objects
        detector_origin (:obj:`numpy array`): The detector origin at s=0, i.e ```geometry_matrix(s=0)[:,0]```.
        geometry_matrix (:obj:`numpy array`): The detector geometry current s, i.e ```geometry_descriptor(s)```.

    """

    def __init__(self, pixel_size, geometry_descriptor ):
        self.pixel_size        = pixel_size
        self.geometry_descriptor   = geometry_descriptor
        self.frames = []
        self.set_geometry(s=0)

    def set_geometry(self, s):
        """Set the geometry of the detector based on the parametric value s in [0,1]

        Args:
            s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
                while s=1 to a beam with wavevector k2. The  geometry matrix will be called for the provided s 

                value and the detector geometry updated.
        """
        self.geometry_matrix = self.geometry_descriptor(s)
        self.detector_origin = self.geometry_matrix[:,0]
        self.normalised_geometry_matrix = self.geometry_matrix / np.linalg.norm(self.geometry_matrix, axis=0)
        self.zdhat, self.zmax  = utils.get_unit_vector_and_l2norm(self.detector_origin, self.geometry_matrix[:,2])
        self.ydhat, self.ymax  = utils.get_unit_vector_and_l2norm(self.detector_origin, self.geometry_matrix[:,1])
        self.normal = np.cross(self.zdhat, self.ydhat)

    def render(self, frame_number):
        """Take a list of scatterers render to a pixelated pattern.

        Args:
            frame_number (:obj:`int`): Index of the frame in the :obj:`frames` list to be rendered.

        Returns:
            A pixelated frame as a (:obj:`numpy array`) with shape infered form the detector geometry and
            pixel size.

        NOTE: This function is meant to alow for overriding when specalised intensity models are to be tested.

        """
        frame = np.zeros( (int(self.zmax/self.pixel_size), int(self.ymax/self.pixel_size)) )
        for scatterer in self.frames[frame_number]:
            self.set_geometry( s = scatterer.s )
            zd, yd = self.get_intersection( scatterer.kprime, scatterer.centroid )
            if self.contains(zd,yd):
                intensity = scatterer.volume * scatterer.lorentz_factor * scatterer.polarization_factor
                if scatterer.structure_factor is not None:
                    intensity = intensity * ( scatterer.real_structure_factor**2 + scatterer.imaginary_structure_factor**2 )
                frame[int(zd/self.pixel_size), int(yd/self.pixel_size)] += intensity
        self.set_geometry( s = 0 )
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
            (:obj:`float`) Cone opening angle divided by two (radians).

        """
        fourth_corner_of_detector = np.expand_dims(self.geometry_matrix[:,2] + (self.geometry_matrix[:,1] - self.detector_origin[:]),axis=1)
        geom_mat = np.concatenate((self.geometry_matrix.copy(), fourth_corner_of_detector), axis=1)
        for i in range(4):
            geom_mat[:,i] -= source_point
        normalised_local_coord_geom_mat = geom_mat/np.linalg.norm(geom_mat, axis=0)
        zd, yd = self.get_intersection(k, source_point)
        cone_opening = np.arccos( np.dot(normalised_local_coord_geom_mat.T, k / np.linalg.norm(k) ) ) # These are two time Bragg angles        
        return np.max( cone_opening ) / 2. 

    def approximate_wrapping_cone(self, beam, samples=180, margin=np.pi/180):
        """Given a moving detector as well as a variable wavevector approximate an upper Bragg angle bound after which scattering
            will not intersect the detector area.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            samples (:obj:`float`): Number of points in s=[0,1] that should be used to approximate the 
                cone opening angle.
            margin (:obj:`float`): Radians added to the returned result (for safety since samples are finite).

        Returns:
            (:obj:`float`) Cone opening angle divided by two (radians).

        NOTE: for specalized use you may override this function and provide some fixed Bragg angle bound

        """
        cone_angles = []
        for s in np.linspace(0, 1, samples):
            self.set_geometry(s)
            beam.set_geometry(s)
            cone_angles.append( self.get_wrapping_cone( beam.k, beam.centroid ) )
        self.set_geometry(s=0)
        beam.set_geometry(s=0)
        return np.max(cone_angles) + margin

