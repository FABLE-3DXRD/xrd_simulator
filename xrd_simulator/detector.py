import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.utils import source
from xrd_simulator import utils
from scipy.interpolate import griddata

class Detector(object):

    """Represents a rectangular X-ray scattering flat area detection device.

    The detector can collect scattering as abstract objects and map them to frame numbers.
    Using a render function these abstract representation can be rendered into pixelated frames.
    The detector is described by a 3 x 3 geometry matrix whic columns contain vectors that attach
    to three of the four corners of the detector.

    Args:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        d0,d1,d2 (:obj:`numpy array`): Detector corner 3d coordinates ```shape=(3,)```. The origin of the detector
            is at d0.

    Attributes:
        pixel_size (:obj:`float`): Pixel side length (square pixels) in units of microns.
        d0,d1,d2 (:obj:`numpy array`): Detector corner 3d coordinates ```shape=(3,)```. The origin of the detector
            is at d0.
        frames (:obj:`list` of :obj:`list` of :obj:`scatterer.Scatterer`): Analytical diffraction patterns which
        zdhat,ydhat (:obj:`numpy array`): Detector basis vectors.
        normal (:obj:`numpy array`): Detector normal.
        zmax,ymax (:obj:`numpy array`): Detector width and height.
    """

    def __init__(self, pixel_size, d0, d1, d2 ):
        self.d0, self.d1, self.d2 = d0, d1, d2
        self.pixel_size = pixel_size
        self.zmax   = np.linalg.norm(self.d2-self.d0)
        self.ymax   = np.linalg.norm(self.d1-self.d0)
        self.zdhat  = (self.d2-self.d0 ) / self.zmax
        self.ydhat  = (self.d1-self.d0 ) / self.ymax
        self.normal = np.cross(self.zdhat, self.ydhat)
        self.frames = []

        zz = np.arange(0, self.zmax, self.pixel_size)
        yy = np.arange(0, self.ymax, self.pixel_size)
        Z,Y = np.meshgrid(zz, yy, indexing='ij')
        self.pixel_real_coordinates = np.array([self.d0 + y*self.ydhat + z*self.zdhat for y,z in zip(Y.flatten(),Z.flatten()) ])
        self.pixel_det_coordinates = np.array([ [z, y] for y,z in zip(Y.flatten(),Z.flatten()) ])

    def render(self, frame_number, lorentz=True, polarization=True, structure_factor=True, method="centroid", verbose=True):
        """Render a pixelated diffraction pattern onto the detector plane .

        Args:
            frame_number (:obj:`int`): Index of the frame in the :obj:`frames` list to be rendered.
            lorentz (:obj:`bool`): Weight scattered intensity by Lorentz factor. Defaults to False.
            polarization (:obj:`bool`): Weight scattered intensity by Polarization factor. Defaults to False.
            structure_factor (:obj:`bool`): Weight scattered intensity by Structure Factor factor. Defaults to False.
            method (:obj:`str`): Rendering method, must be one of ```project``` or ```centroid```. Defaults to ```centroid```.
                The default,```method=centroid```, is a simple deposit of intensity for each scatterer onto the detector by
                tracing a line from the sample scattering region centroid to the detector plane. The intensity is deposited
                into a single detector pixel regardless of the geometrical shape of the scatterer. If instead ```method=project```
                the scattering regions are projected onto the detector depositing a intensity over possibly several pixels as 
                weighted by the optical path lengths of the rays diffracting from the scattering region.
            verbose (:obj:`bool`): Prints progress. Defaults to True.

        Returns:
            A pixelated frame as a (:obj:`numpy array`) with shape infered form the detector geometry and
            pixel size.

        NOTE: This function can be overwitten to do more advanced models for intensity.
        """
        frame = np.zeros( (int(self.zmax/self.pixel_size), int(self.ymax/self.pixel_size)) )
        for si,scatterer in enumerate(self.frames[frame_number]):

            if verbose:
                progress_bar_message = "Rendering "+str(len(self.frames[frame_number]))+" scattering volumes unto the detector"
                progress_fraction    = float(si+1)/len(self.frames[frame_number])
                utils.print_progress(progress_fraction, message=progress_bar_message)

            if method=='project':
                self._projection_render(scatterer, frame, lorentz, polarization, structure_factor)
            elif  method=='centroid':
                self._centroid_render(scatterer, frame, lorentz, polarization, structure_factor)

        return frame

    def _centroid_render(self, scatterer, frame, lorentz, polarization, structure_factor):
        """Simple deposit of intensity for each scatterer onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited into a single
        detector pixel regardless of the geometrical shape of the scatterer.
        """
        zd, yd = self.get_intersection( scatterer.scattered_wave_vector, scatterer.centroid )
        if self.contains(zd,yd): 
            intensity_scaling_factor = self._get_intensity_factor( scatterer, lorentz, polarization, structure_factor )
            frame[int(zd/self.pixel_size), int(yd/self.pixel_size)] += scatterer.volume * intensity_scaling_factor

    def _projection_render(self, scatterer, frame, lorentz, polarization, structure_factor):
        """Raytrace and project the scattering regions onto the detector plane for increased peak shape accuracy.
        This is generally very computationally expensive compared to the simpler (:obj:`function`):render function.
        """
        mask, projection = self._project( scatterer )
        if projection is not None:
            intensity_scaling_factor = self._get_intensity_factor( scatterer, lorentz, polarization, structure_factor )
            frame[ mask.reshape(frame.shape) ] += projection * intensity_scaling_factor

    def _get_intensity_factor(self, scatterer, lorentz, polarization, structure_factor):
        intensity_factor = 1.0
        if lorentz:
            intensity_factor *= scatterer.lorentz_factor 
        if polarization:
            intensity_factor *= scatterer.polarization_factor
        if structure_factor and scatterer.real_structure_factor is not None:
            intensity_factor *= ( scatterer.real_structure_factor**2 + scatterer.imaginary_structure_factor**2 )
        return intensity_factor

    def _project( self, scatterer ):
        """Compute parametric projection of scattering region unto detector.

            NOTE: Mike Cyrus and Jay Beck. “Generalized two- and three-dimensional clipping”. (1978)
            (based on orthogonal equations: (p - e - t*r) . n = 0 )
        """
        # TODO: document and cleanup, can the code be shorter, something moved to utils?
        ray_direction = scatterer.scattered_wave_vector / np.linalg.norm( scatterer.scattered_wave_vector )

        vertices = scatterer.convex_hull.points[ scatterer.convex_hull.vertices ]
        vp = np.array( [self.get_intersection( ray_direction, v) for v in vertices] )

        minzd, maxzd = np.min(vp[:,0]), np.max(vp[:,0])
        minyd, maxyd = np.min(vp[:,1]), np.max(vp[:,1])
        mask = (self.pixel_det_coordinates[:,1]<=maxyd)*(self.pixel_det_coordinates[:,1]>=minyd)*(self.pixel_det_coordinates[:,0]<=maxzd)*(self.pixel_det_coordinates[:,0]>=minzd)
        
        if np.sum(mask)==0: 
            return None,None
        else:
            ray_points = self.pixel_real_coordinates[mask,:]

            plane_normals = scatterer.convex_hull.equations[:,0:3]
            plane_ofsets  = scatterer.convex_hull.equations[:,3].reshape(scatterer.convex_hull.equations.shape[0], 1)
            plane_points  = -np.multiply( plane_ofsets, plane_normals ) 

            ray_points    = np.ascontiguousarray(ray_points)
            ray_direction = np.ascontiguousarray(ray_direction)
            plane_points  = np.ascontiguousarray(plane_points)
            plane_normals = np.ascontiguousarray(plane_normals)

            clip_lengths  = utils.clip_line_with_convex_polyhedron(ray_points, ray_direction, plane_points, plane_normals)

        return mask, clip_lengths

    def get_intersection(self, ray_direction, source_point):
        """Get detector intersection in detector coordinates of singel a ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vector in direction of the X-ray propagation 
            source_point (:obj:`numpy array`): Origin of the ray.

        Returns:
            (:obj:`tuple`) zd, yd in detector plane coordinates.

        """
        #TODO: Consider moving this to utils.py and generalise for line and plane
        s = (self.d0 - source_point).dot(self.normal) / ray_direction.dot(self.normal)
        intersection = source_point + ray_direction*s
        zd = np.dot( intersection - self.d0 , self.zdhat)
        yd = np.dot( intersection - self.d0 , self.ydhat)
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
            k (:obj:`numpy array`): Wavevector forming the central axis of cone ´´´shape=(3,)´´´.
            source_point (:obj:`numpy array`): Origin of the wavevector ´´´shape=(3,)´´´.

        Returns:
            (:obj:`float`) Cone opening angle divided by two (radians), corresponding to a maximum bragg angle after
                which scattering will systematically miss the detector.

        """
        fourth_corner_of_detector = self.d2 + (self.d1 - self.d0[:])
        geom_mat = np.zeros((3,4))
        for i,det_corner in enumerate([self.d0,self.d1,self.d2,fourth_corner_of_detector]):
            geom_mat[:,i] = det_corner - source_point
        normalised_local_coord_geom_mat = geom_mat/np.linalg.norm(geom_mat, axis=0)
        cone_opening = np.arccos( np.dot(normalised_local_coord_geom_mat.T, k / np.linalg.norm(k) ) ) # These are two time Bragg angles        
        return np.max( cone_opening ) / 2.

