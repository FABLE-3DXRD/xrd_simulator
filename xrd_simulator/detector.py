import numpy as np
from xrd_simulator import utils
from xrd_simulator._pickleable_object import PickleableObject


class Detector(PickleableObject):

    """Represents a rectangular X-ray scattering flat area detection device.

    The detector can collect scattering as abstract objects and map them to frame numbers.
    Using a render function these abstract representation can be rendered into pixelated frames.
    The detector is described by a 3 x 3 geometry matrix which columns contain vectors that attach
    to three of the four corners of the detector.

    Args:
        pixel_size_z (:obj:`float`): Pixel side length along zdhat (rectangular pixels) in units of microns.
        pixel_size_y (:obj:`float`): Pixel side length along ydhat (rectangular pixels) in units of microns.
        det_corner_0,det_corner_1,det_corner_2 (:obj:`numpy array`): Detector corner 3d coordinates ``shape=(3,)``.
            The origin of the detector is at det_corner_0.

    Attributes:
        pixel_size_z (:obj:`float`): Pixel side length along zdhat (rectangular pixels) in units of microns.
        pixel_size_y (:obj:`float`): Pixel side length along ydhat (rectangular pixels) in units of microns.
        det_corner_0,det_corner_1,det_corner_2 (:obj:`numpy array`): Detector corner 3d coordinates ``shape=(3,)``.
            The origin of the detector is at det_corner_0.
        frames (:obj:`list` of :obj:`list` of :obj:`scatterer.Scatterer`): Analytical diffraction patterns which
        zdhat,ydhat (:obj:`numpy array`): Detector basis vectors.
        normal (:obj:`numpy array`): Detector normal.
        zmax,ymax (:obj:`numpy array`): Detector width and height.
        pixel_coordinates  (:obj:`numpy array`): Real space 3d detector pixel coordinates. ``shape=(n,3)``

    Examples:
        .. literalinclude:: examples/example_init_detector.py

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
                The default,```method=centroid```, is a simple deposit of intensity for each scatterer onto the detector by
                tracing a line from the sample scattering region centroid to the detector plane. The intensity is deposited
                into a single detector pixel regardless of the geometrical shape of the scatterer. If instead
                ```method=project``` the scattering regions are projected onto the detector depositing a intensity over
                possibly several pixels as weighted by the optical path lengths of the rays diffracting from the scattering
                region.
            verbose (:obj:`bool`): Prints progress. Defaults to True.

        Returns:
            A pixelated frame as a (:obj:`numpy array`) with shape inferred form the detector geometry and
            pixel size.

        NOTE: This function can be overwitten to do more advanced models for intensity.

        """
        frame = np.zeros((int(self.zmax / self.pixel_size_z),
                         int(self.ymax / self.pixel_size_y)))
        for si, scatterer in enumerate(self.frames[frame_number]):

            if verbose:
                progress_bar_message = "Rendering " + \
                    str(len(self.frames[frame_number])) + " scattering volumes unto the detector"
                progress_fraction = float(
                    si + 1) / len(self.frames[frame_number])
                utils.print_progress(
                    progress_fraction,
                    message=progress_bar_message)

            if method == 'project':
                self._projection_render(
                    scatterer, frame, lorentz, polarization, structure_factor)
            elif method == 'centroid':
                self._centroid_render(
                    scatterer,
                    frame,
                    lorentz,
                    polarization,
                    structure_factor)

        return frame

    def get_intersection(self, ray_direction, source_point):
        """Get detector intersection in detector coordinates of a single ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vector in direction of the X-ray propagation
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

    def get_wrapping_cone(self, k, source_point):
        """Compute the cone around a wavevector such that the cone wraps the detector corners.

        Args:
            k (:obj:`numpy array`): Wavevector forming the central axis of cone ´´´shape=(3,)´´´.
            source_point (:obj:`numpy array`): Origin of the wavevector ´´´shape=(3,)´´´.

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

    def _get_pixel_coordinates(self):
        zds = np.arange(0, self.zmax, self.pixel_size_z)
        yds = np.arange(0, self.ymax, self.pixel_size_y)
        pixel_coordinates = np.zeros((len(zds), len(yds), 3))
        for i, z in enumerate(zds):
            for j, y in enumerate(yds):
                pixel_coordinates[i, j, :] = self.det_corner_0 + \
                    y * self.ydhat + z * self.zdhat
        return pixel_coordinates

    def _centroid_render(
            self,
            scatterer,
            frame,
            lorentz,
            polarization,
            structure_factor):
        """Simple deposit of intensity for each scatterer onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited into a single
        detector pixel regardless of the geometrical shape of the scatterer.
        """
        zd, yd = self.get_intersection(
            scatterer.scattered_wave_vector, scatterer.centroid)
        if self.contains(zd, yd):
            intensity_scaling_factor = self._get_intensity_factor(
                scatterer, lorentz, polarization, structure_factor)
            row, col = self._detector_coordinate_to_pixel_index(zd, yd)
            frame[row, col] += scatterer.volume * intensity_scaling_factor

    def _projection_render(
            self,
            scatterer,
            frame,
            lorentz,
            polarization,
            structure_factor):
        """Raytrace and project the scattering regions onto the detector plane for increased peak shape accuracy.
        This is generally very computationally expensive compared to the simpler (:obj:`function`):_centroid_render
        function.

        NOTE: If the projection of the scatterer does not hit any pixel centroids of the detector fallback to
        the (:obj:`function`):_centroid_render function is used to deposit the intensity into a single detector pixel.
        """
        box = self._get_projected_bounding_box(scatterer)
        if box is not None:
            projection = self._project(scatterer, box)
            if np.sum(projection) == 0:
                # The projection of the scatterer did not hit any pixel centroids of the detector.
                # i.e the scatterer is small in comparison to the detector
                # pixels.
                self._centroid_render(
                    scatterer,
                    frame,
                    lorentz,
                    polarization,
                    structure_factor)
            else:
                intensity_scaling_factor = self._get_intensity_factor(
                    scatterer, lorentz, polarization, structure_factor)
                frame[box[0]:box[1], box[2]:box[3]] += projection * \
                    intensity_scaling_factor * self.pixel_size_z * self.pixel_size_y

    def _get_intensity_factor(
            self,
            scatterer,
            lorentz,
            polarization,
            structure_factor):
        intensity_factor = 1.0
        # TODO: Consider solid angle intensity rescaling.
        if lorentz:
            intensity_factor *= scatterer.lorentz_factor
        if polarization:
            intensity_factor *= scatterer.polarization_factor
        if structure_factor and scatterer.real_structure_factor is not None:
            intensity_factor *= (scatterer.real_structure_factor **
                                 2 + scatterer.imaginary_structure_factor**2)
        return intensity_factor

    def _project(self, scatterer, box):
        """Compute parametric projection of scattering region unto detector.
        """

        ray_points = self.pixel_coordinates[box[0]:box[1], box[2]:box[3], :].reshape(
            (box[1] - box[0]) * (box[3] - box[2]), 3)

        plane_normals = scatterer.convex_hull.equations[:, 0:3]
        plane_ofsets = scatterer.convex_hull.equations[:, 3].reshape(
            scatterer.convex_hull.equations.shape[0], 1)
        plane_points = -np.multiply(plane_ofsets, plane_normals)

        ray_points = np.ascontiguousarray(ray_points)
        ray_direction = np.ascontiguousarray(
            scatterer.scattered_wave_vector /
            np.linalg.norm(
                scatterer.scattered_wave_vector))
        plane_points = np.ascontiguousarray(plane_points)
        plane_normals = np.ascontiguousarray(plane_normals)

        clip_lengths = utils.clip_line_with_convex_polyhedron(
            ray_points, ray_direction, plane_points, plane_normals)
        clip_lengths = clip_lengths.reshape(box[1] - box[0], box[3] - box[2])

        return clip_lengths

    def _detector_coordinate_to_pixel_index(self, zd, yd):
        row_index = int(zd / self.pixel_size_z)
        col_index = int(yd / self.pixel_size_y)
        return row_index, col_index

    def _get_projected_bounding_box(self, scatterer):
        """Compute bounding detector pixel indices of the bounding the projection of a scattering region.

        Args:
            scatterer (:obj:`xrd_simulator.Scatterer`): The scattering region.

        Returns:
            (:obj:`float`) indices that can be used to slice the detector frame array and get the pixels that
                are within the bounding box.

        """
        vertices = scatterer.convex_hull.points[scatterer.convex_hull.vertices]
        projected_vertices = np.array([self.get_intersection(
            scatterer.scattered_wave_vector, v) for v in vertices])

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
