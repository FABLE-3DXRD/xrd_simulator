"""The detector module is used to represent a 2D area detector. After diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` has been computed, the detector can render
the scattering as a pixelated image via the :func:`xrd_simulator.detector.Detector.render`
function.

Here is a minimal example of how to instantiate a detector object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_detector.py

Below follows a detailed description of the detector class attributes and functions.

"""
import numpy as np
from xrd_simulator import utils
import dill
from scipy.signal import convolve


class Detector():

    """Represents a rectangular 2D area detector.

    The detector collects :class:`xrd_simulator.scattering_unit.ScatteringUnit` during diffraction from a
    :class:`xrd_simulator.polycrystal.Polycrystal`. The detector implements various rendering of the scattering
    as 2D pixelated images. The detector geometry is described by specifying the locations of three detector
    corners. The detector is described in the laboratory coordinate system.

    Args:
        pixel_size_z (:obj:`float`): Pixel side length along zdhat (rectangular pixels) in units of microns.
            (zdhat is the unit vector from det_corner_0 towards det_corner_2)
        pixel_size_y (:obj:`float`): Pixel side length along ydhat (rectangular pixels) in units of microns.
            (ydhat is the unit vector from det_corner_0 towards det_corner_1)
        det_corner_0,det_corner_1,det_corner_2 (:obj:`numpy array`): Detector corner 3d coordinates ``shape=(3,)``.
            The origin of the detector is at det_corner_0.

    Attributes:
        pixel_size_z (:obj:`float`): Pixel side length along zdhat (rectangular pixels) in units of microns.
        pixel_size_y (:obj:`float`): Pixel side length along ydhat (rectangular pixels) in units of microns.
        det_corner_0,det_corner_1,det_corner_2 (:obj:`numpy array`): Detector corner 3d coordinates ``shape=(3,)``.
            The origin of the detector is at det_corner_0.
        frames (:obj:`list` of :obj:`list` of :obj:`scattering_unit.ScatteringUnit`): Analytical diffraction patterns which
        zdhat,ydhat (:obj:`numpy array`): Detector basis vectors.
        normal (:obj:`numpy array`): Detector normal, fromed as the cross product: numpy.cross(self.zdhat, self.ydhat)
        zmax,ymax (:obj:`numpy array`): Detector width and height.
        pixel_coordinates  (:obj:`numpy array`): Real space 3d detector pixel coordinates. ``shape=(n,3)``
        point_spread_function  (:obj:`callable`): Scalar point spread function called as point_spread_function(z, y). The
            z and y coordinates are assumed to be in local units of pixels. I.e point_spread_function(0, 0) returns the
            value of the pointspread function at the location of the point being spread. This is meant to model blurring
            due to detector optics. Defaults to a Gaussian with standard deviation 1.0 and mean at (z,y)=(0,0).


    """

    def __init__(self, pixel_size_z, pixel_size_y, det_corner_0, det_corner_1, det_corner_2):
        self.det_corner_0, self.det_corner_1, self.det_corner_2 = det_corner_0, det_corner_1, det_corner_2
        self.pixel_size_z = pixel_size_z
        self.pixel_size_y = pixel_size_y
        self.zmax = np.linalg.norm(det_corner_2 - det_corner_0)
        self.ymax = np.linalg.norm(det_corner_1 - det_corner_0)
        self.zdhat = (det_corner_2 - det_corner_0) / self.zmax
        self.ydhat = (det_corner_1 - det_corner_0) / self.ymax
        self.normal = np.cross(self.zdhat, self.ydhat)
        self.frames = []
        self.pixel_coordinates = self._get_pixel_coordinates()

        self.point_spread_function = lambda z, y: np.exp(-0.5 * (z*z + y*y) / (1.0 * 1.0) )
        self._point_spread_kernel_shape = (5, 5)

    @property
    def point_spread_kernel_shape(self):
        """point_spread_kernel_shape  (:obj:`tuple`): Number of pixels in zdhat and ydhat over which to apply the pointspread
            function for each scattering event. I.e the shape of the kernel that will be convolved with the diffraction
            pattern. The values of the kernel is defined by the point_spread_function. Defaults to shape (5, 5).

            NOTE: The point_spread_function is automatically normalised over the point_spread_kernel_shape domain such that the
                final convolution over the detector diffraction pattern is intensity preserving.
        """
        return self._point_spread_kernel_shape

    @point_spread_kernel_shape.setter
    def point_spread_kernel_shape(self, kernel_shape):

        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise ValueError("Point spread function kernel shape must be odd in both dimensions, but shape "+str(kernel_shape)+" was provided.")
        else:
            self._point_spread_kernel_shape = kernel_shape

    def render(
            self,
            frame_number,
            lorentz=True,
            polarization=True,
            structure_factor=True,
            method="centroid",
            verbose=True):
        """Render a pixelated diffraction pattern onto the detector plane .

        NOTE: The value read out on a pixel in the detector is an approximation of the integrated number of counts over
        the pixel area.

        Args:
            frame_number (:obj:`int`): Index of the frame in the :obj:`frames` list to be rendered.
            lorentz (:obj:`bool`): Weight scattered intensity by Lorentz factor. Defaults to False.
            polarization (:obj:`bool`): Weight scattered intensity by Polarization factor. Defaults to False.
            structure_factor (:obj:`bool`): Weight scattered intensity by Structure Factor factor. Defaults to False.
            method (:obj:`str`): Rendering method, must be one of ```project``` or ```centroid```. Defaults to ```centroid```.
                The default,```method=centroid```, is a simple deposit of intensity for each scattering_unit onto the detector by
                tracing a line from the sample scattering region centroid to the detector plane. The intensity is deposited
                into a single detector pixel regardless of the geometrical shape of the scattering_unit. If instead
                ```method=project``` the scattering regions are projected onto the detector depositing a intensity over
                possibly several pixels as weighted by the optical path lengths of the rays diffracting from the scattering
                region.
            verbose (:obj:`bool`): Prints progress. Defaults to True.

        Returns:
            A pixelated frame as a (:obj:`numpy array`) with shape inferred form the detector geometry and
            pixel size.

        NOTE: This function can be overwitten to do more advanced models for intensity.

        """
        frame = np.zeros( (self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]) )

        for si, scattering_unit in enumerate(self.frames[frame_number]):

            if verbose:
                progress_bar_message = "Rendering " + \
                    str(len(self.frames[frame_number])) + " scattering volumes unto the detector"
                progress_fraction = float(
                    si + 1) / len(self.frames[frame_number])
                utils._print_progress(
                    progress_fraction,
                    message=progress_bar_message)

            if method == 'project':
                self._projection_render(
                    scattering_unit, frame, lorentz, polarization, structure_factor)
            elif method == 'centroid':
                self._centroid_render(
                    scattering_unit,
                    frame,
                    lorentz,
                    polarization,
                    structure_factor)

        if self.point_spread_function is not None:
            kernel = self._get_point_spread_function_kernel()
            frame = convolve(frame, kernel, mode='same', method='direct')

        return frame

    def get_intersection(self, ray_direction, source_point):
        """Get detector intersection in detector coordinates of a single ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vector in direction of the xray propagation
            source_point (:obj:`numpy array`): Origin of the ray.

        Returns:
            (:obj:`tuple`) zd, yd in detector plane coordinates.

        """
        s = (self.det_corner_0 - source_point).dot(self.normal) / \
            ray_direction.dot(self.normal)
        intersection = source_point + ray_direction * s
        zd = np.dot(intersection - self.det_corner_0, self.zdhat)
        yd = np.dot(intersection - self.det_corner_0, self.ydhat)
        return zd, yd

    def contains(self, zd, yd):
        """Determine if the detector coordinate zd,yd lies within the detector bounds.

        Args:
            zd (:obj:`float`): Detector z coordinate
            yd (:obj:`float`): Detector y coordinate

        Returns:
            (:obj:`boolean`) True if the zd,yd is within the detector bounds.

        """
        return zd >= 0 and zd <= self.zmax and yd >= 0 and yd <= self.ymax

    def project(self, scattering_unit, box):
        """Compute parametric projection of scattering region unto detector.

        Args:
            scattering_unit (:obj:`xrd_simulator.ScatteringUnit`): The scattering region.
            box (:obj:`tuple` of :obj:`int`): indices of the detector frame over which to compute the projection.
                i.e the subgrid of the detector is taken as: array[[box[0]:box[1], box[2]:box[3]].

        Returns:
            (:obj:`numpy array`) clip lengths between scattering_unit polyhedron and rays traced from the detector.

        """

        ray_points = self.pixel_coordinates[box[0]:box[1], box[2]:box[3], :].reshape(
            (box[1] - box[0]) * (box[3] - box[2]), 3)

        plane_normals = scattering_unit.convex_hull.equations[:, 0:3]
        plane_ofsets = scattering_unit.convex_hull.equations[:, 3].reshape(
            scattering_unit.convex_hull.equations.shape[0], 1)
        plane_points = -np.multiply(plane_ofsets, plane_normals)

        ray_points = np.ascontiguousarray(ray_points)
        ray_direction = np.ascontiguousarray(
            scattering_unit.scattered_wave_vector /
            np.linalg.norm(
                scattering_unit.scattered_wave_vector))
        plane_points = np.ascontiguousarray(plane_points)
        plane_normals = np.ascontiguousarray(plane_normals)

        clip_lengths = utils._clip_line_with_convex_polyhedron(
            ray_points, ray_direction, plane_points, plane_normals)
        clip_lengths = clip_lengths.reshape(box[1] - box[0], box[3] - box[2])

        return clip_lengths

    def get_wrapping_cone(self, k, source_point):
        """Compute the cone around a wavevector such that the cone wraps the detector corners.

        Args:
            k (:obj:`numpy array`): Wavevector forming the central axis of cone ```shape=(3,)```.
            source_point (:obj:`numpy array`): Origin of the wavevector ```shape=(3,)```.

        Returns:
            (:obj:`float`) Cone opening angle divided by two (radians), corresponding to a maximum bragg angle after
                which scattering will systematically miss the detector.

        """
        fourth_corner_of_detector = self.det_corner_2 + (self.det_corner_1 - self.det_corner_0[:])
        geom_mat = np.zeros((3, 4))
        for i, det_corner in enumerate(
                [self.det_corner_0, self.det_corner_1, self.det_corner_2, fourth_corner_of_detector]):
            geom_mat[:, i] = det_corner - source_point
        normalised_local_coord_geom_mat = geom_mat / \
            np.linalg.norm(geom_mat, axis=0)
        cone_opening = np.arccos(
            np.dot(
                normalised_local_coord_geom_mat.T,
                k / np.linalg.norm(k)))  # These are two time Bragg angles
        return np.max(cone_opening) / 2.

    def save(self, path):
        """Save the detector object to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.

        """
        if not path.endswith(".det"):
            path = path + ".det"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the detector object from disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        if not path.endswith(".det"):
            raise ValueError("The loaded motion file must end with .det")
        with open(path, 'rb') as f:
            return dill.load(f)

    def _get_point_spread_function_kernel(self):
        """Render the point_spread_function onto a grid of shape specified by point_spread_kernel_shape.
        """
        sz, sy = self.point_spread_kernel_shape
        axz = np.linspace(-(sz - 1) / 2., (sz - 1) / 2., sz)
        axy = np.linspace(-(sy - 1) / 2., (sy - 1) / 2., sy)
        Z, Y = np.meshgrid(axz, axy, indexing='ij')
        kernel = np.zeros(self.point_spread_kernel_shape)
        for i in range(Z.shape[0]):
            for j in range(Y.shape[1]):
                kernel[i, j] = self.point_spread_function(Z[i, j], Y[i, j])

        assert len( kernel[kernel<0] )==0, "Point spread function must be strictly positive, but negative values were found."
        assert np.sum(kernel)>1e-8, "The integrated value of the point spread function over the defined kernel domain is close to zero."

        return kernel / np.sum(kernel)

    def _get_pixel_coordinates(self):
        zds = np.arange(0, self.zmax, self.pixel_size_z)
        yds = np.arange(0, self.ymax, self.pixel_size_y)
        Z, Y = np.meshgrid(zds, yds, indexing='ij')
        Zds = np.zeros((len(zds), len(yds), 3))
        Yds = np.zeros((len(zds), len(yds), 3))
        for i in range(3):
            Zds[:,:,i] = Z
            Yds[:,:,i] = Y
        pixel_coordinates = self.det_corner_0.reshape(1,1,3) + Zds*self.zdhat.reshape(1,1,3) + Yds*self.ydhat.reshape(1,1,3)
        return pixel_coordinates

    def _centroid_render(
            self,
            scattering_unit,
            frame,
            lorentz,
            polarization,
            structure_factor):
        """Simple deposit of intensity for each scattering_unit onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited into a single
        detector pixel regardless of the geometrical shape of the scattering_unit.
        """
        zd, yd = self.get_intersection(
            scattering_unit.scattered_wave_vector, scattering_unit.centroid)
        if self.contains(zd, yd):
            intensity_scaling_factor = self._get_intensity_factor(
                scattering_unit, lorentz, polarization, structure_factor)
            row, col = self._detector_coordinate_to_pixel_index(zd, yd)
            frame[row, col] += scattering_unit.volume * intensity_scaling_factor

    def _projection_render(
            self,
            scattering_unit,
            frame,
            lorentz,
            polarization,
            structure_factor):
        """Raytrace and project the scattering regions onto the detector plane for increased peak shape accuracy.
        This is generally very computationally expensive compared to the simpler (:obj:`function`):_centroid_render
        function.

        NOTE: If the projection of the scattering_unit does not hit any pixel centroids of the detector fallback to
        the (:obj:`function`):_centroid_render function is used to deposit the intensity into a single detector pixel.
        """
        box = self._get_projected_bounding_box(scattering_unit)
        if box is not None:
            projection = self.project(scattering_unit, box)
            if np.sum(projection) == 0:
                # The projection of the scattering_unit did not hit any pixel centroids of the detector.
                # i.e the scattering_unit is small in comparison to the detector
                # pixels.
                self._centroid_render(
                    scattering_unit,
                    frame,
                    lorentz,
                    polarization,
                    structure_factor)
            else:
                intensity_scaling_factor = self._get_intensity_factor(
                    scattering_unit, lorentz, polarization, structure_factor)
                frame[box[0]:box[1], box[2]:box[3]] += projection * \
                    intensity_scaling_factor * self.pixel_size_z * self.pixel_size_y

    def _get_intensity_factor(
            self,
            scattering_unit,
            lorentz,
            polarization,
            structure_factor):
        intensity_factor = 1.0
        # TODO: Consider solid angle intensity rescaling and air scattering.
        if lorentz:
            intensity_factor *= scattering_unit.lorentz_factor
        if polarization:
            intensity_factor *= scattering_unit.polarization_factor
        if structure_factor:
            if scattering_unit.phase.structure_factors is not None:
                intensity_factor *= (scattering_unit.real_structure_factor **
                                    2 + scattering_unit.imaginary_structure_factor**2)
            else:
                raise ValueError("Structure factors have not been set, .cif file is required at sample instantiation.")

        return intensity_factor

    def _detector_coordinate_to_pixel_index(self, zd, yd):
        row_index = int(zd / self.pixel_size_z)
        col_index = int(yd / self.pixel_size_y)
        return row_index, col_index

    def _get_projected_bounding_box(self, scattering_unit):
        """Compute bounding detector pixel indices of the bounding the projection of a scattering region.

        Args:
            scattering_unit (:obj:`xrd_simulator.ScatteringUnit`): The scattering region.

        Returns:
            (:obj:`tuple` of :obj:`int`) indices that can be used to slice the detector frame array and get the pixels that
                are within the bounding box.

        """
        vertices = scattering_unit.convex_hull.points[scattering_unit.convex_hull.vertices]
        projected_vertices = np.array([self.get_intersection(
            scattering_unit.scattered_wave_vector, v) for v in vertices])

        min_zd, max_zd = np.min(projected_vertices[:, 0]), np.max(
            projected_vertices[:, 0])
        min_yd, max_yd = np.min(projected_vertices[:, 1]), np.max(
            projected_vertices[:, 1])

        min_zd, max_zd = np.max([min_zd, 0]), np.min([max_zd, self.zmax])
        min_yd, max_yd = np.max([min_yd, 0]), np.min([max_yd, self.ymax])

        if min_zd > max_zd or min_yd > max_yd:
            return None

        min_row_indx, min_col_indx = self._detector_coordinate_to_pixel_index(
            min_zd, min_yd)
        max_row_indx, max_col_indx = self._detector_coordinate_to_pixel_index(
            max_zd, max_yd)

        max_row_indx = np.min(
            [max_row_indx + 1, int(self.zmax / self.pixel_size_z)])
        max_col_indx = np.min(
            [max_col_indx + 1, int(self.ymax / self.pixel_size_y)])

        return min_row_indx, max_row_indx, min_col_indx, max_col_indx
