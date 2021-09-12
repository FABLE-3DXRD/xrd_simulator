import numpy as np 
import matplotlib.pyplot as plt
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
        geometry_matrix (:obj:`callable`): geometry_matrix(s) <- G where G is a ```shape=(3,3)``` geometry
            matrix which columns attach to three corners of the detector array (units of microns).

    Attributes:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        geometry_matrix (:obj:`callable`): geometry_matrix(s) <- G where G is a ```shape=(3,3)``` geometry
            matrix which columns attach to three corners of the detector array (units of microns).
        frames (:obj:`list` of :obj:`list` of :obj:`scatterer.Scatterer`): Analytical diffraction patterns which
            may be rendered into pixelated frames. Each frame is a list of scattering objects

    """

    def __init__(self, pixel_size, geometry_matrix ):
        self.pixel_size        = pixel_size
        self.geometry_matrix   = geometry_matrix
        self.frames = []
        self.set_geometry(s=0)

    def set_geometry(self, s):
        """Set the geometry of the detector based on the parametric value s in [0,1]

        Args:
            s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
                while s=1 to a beam with wavevector k2. The  geometry matrix will be called for the provided s 

                value and the detector geometry updated.
        """
        G = self.geometry_matrix(s)
        self.normalised_geometry_matrix = G / np.linalg.norm(G, axis=0)
        self.zdhat, self.zmax  = utils.get_unit_vector_and_l2norm(G[:,2], G[:,0])
        self.ydhat, self.ymax  = utils.get_unit_vector_and_l2norm(G[:,1], G[:,0])
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
        """Get detector intersection in detector coordinates of singel a ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vector in direction of the Z.ray propagation 
            source_point (:obj:`numpy array`): 

        Returns:
            (:obj:`tuple`) zd, yd in detector plane coordinates.

        """
        s = ( self.zdhat.dot(self.normal) - source_point.dot(self.normal) ) / ray_direction.dot(self.normal)
        det_intersection =  source_point + ray_direction*s - self.geometry_matrix[0,:]
        zd = np.dot(det_intersection, self.zdhat)
        yd = np.dot(det_intersection, self.ydhat)
        return zd, yd

    def contains(self, zd, yd):
        """Determine if the detector coordinate zd,yd lies within the detector bounds.

        Args:
            zd (:obj:`float`): Detector z coordinate
            yd (:obj:`float`): Detector y coordinate

        Returns:
            (:obj:`boolean`) True if the zd,yd is within the detector bounds.

        """
        return zd>=0 and zd<=self.zmax and yd>=0 and yd<=self.zmax

    def get_wrapping_cone(self, k):
        """Compute the cone around a wavevector such that the cone intersects one detector corner.

        Args:
            k (:obj:`numpy array`): Wavevector forming the central axis of cone.

        Returns:
            (:obj:`float`) Cone opening angle divided by two (radians).

        """
        #TODO: Verify that the min cone openings occur at k1 or k2 
        zd, yd = self.get_intersection(k, c=0)
        assert self.contains(zd, yd), "You provided a wavevector, k="+str(k)+", that will not intersect the detector."
        cone_opening = np.arccos( np.dot(self.normalised_geometry_matrix.T, k / np.linalg.norm(k) ) ) # These are two time Bragg angles
        return np.min( cone_opening ) / 2. 

    def approximate_wrapping_cone(self, beam, samples=180, margin=np.pi/180):
        """Given a moving detector as well as a variable wavevector approximate an upper Bragg angle bound.

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
            cone_angles.append( self.get_wrapping_cone( beam.k ) )
        self.set_geometry(s=0)
        beam.set_geometry(s=0)
        return np.min(cone_angles) + margin

