"""The detector module is used to represent a 2D area detector. After diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` has been computed, the detector can render
the scattering as a pixelated image via the :func:`xrd_simulator.detector.Detector.render`
function.

Here is a minimal example of how to instantiate a detector object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_detector.py

Below follows a detailed description of the detector class attributes and functions.

"""

import xrd_simulator.cuda
import numpy as np
from xrd_simulator import utils
from xrd_simulator.utils import ensure_torch
import dill
import torch

torch.set_default_dtype(torch.float64)


class Detector:
    """Represents a rectangular 2D area detector.

    The detector collects :class:`xrd_simulator.scattering_unit.ScatteringUnit` during diffraction from a
    :class:`xrd_simulator.polycrystal.Polycrystal`. The detector implements various rendering of the scattering
    as 2D pixelated frames. The detector geometry is described by specifying the locations of three detector
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
        frames (:obj:`list` of :obj:`list` of :obj:`scattering_unit.ScatteringUnit`): Analytical diffraction frames which
        zdhat,ydhat (:obj:`numpy array`): Detector basis vectors.
        normal (:obj:`numpy array`): Detector normal, fromed as the cross product: numpy.cross(self.zdhat, self.ydhat)
        zmax,ymax (:obj:`numpy array`): Detector width and height.
        pixel_coordinates  (:obj:`numpy array`): Real space 3d detector pixel coordinates. ``shape=(n,3)``
        point_spread_function  (:obj:`callable`): Scalar point spread function called as point_spread_function(z, y). The
            z and y coordinates are assumed to be in local units of pixels. I.e point_spread_function(0, 0) returns the
            value of the pointspread function at the location of the point being spread. This is meant to model blurring
            due to detector optics. Defaults to a Gaussian with standard deviation 1.0 and mean at (z,y)=(0,0).


    """

    def __init__(
        self, pixel_size_z, pixel_size_y, det_corner_0, det_corner_1, det_corner_2
    ):
        self.det_corner_0 = ensure_torch(det_corner_0)
        self.det_corner_1 = ensure_torch(det_corner_1)
        self.det_corner_2 = ensure_torch(det_corner_2)

        self.pixel_size_z = ensure_torch(pixel_size_z)
        self.pixel_size_y = ensure_torch(pixel_size_y)

        self.zmax = torch.linalg.norm(self.det_corner_2 - self.det_corner_0)
        self.ymax = torch.linalg.norm(self.det_corner_1 - self.det_corner_0)

        self.zdhat = (self.det_corner_2 - self.det_corner_0) / self.zmax
        self.ydhat = (self.det_corner_1 - self.det_corner_0) / self.ymax
        self.normal = torch.linalg.cross(self.zdhat, self.ydhat)
        self.normal = self.normal / torch.linalg.norm(self.normal)
        self.frames = []
        self.pixel_coordinates = self._get_pixel_coordinates()
        self._point_spread_kernel_shape = (5, 5)

    def point_spread_function(self, z, y):
        return np.exp(-0.5 * (z * z + y * y) / (1.0 * 1.0))

    @property
    def point_spread_kernel_shape(self):
        """point_spread_kernel_shape  (:obj:`tuple`): Number of pixels in zdhat and ydhat over which to apply the pointspread
        function for each scattering event. I.e the shape of the kernel that will be convolved with the diffraction
        frame. The values of the kernel is defined by the point_spread_function. Defaults to shape (5, 5).

        NOTE: The point_spread_function is automatically normalised over the point_spread_kernel_shape domain such that the
            final convolution over the detector diffraction frame is intensity preserving.
        """
        return self._point_spread_kernel_shape

    @point_spread_kernel_shape.setter
    def point_spread_kernel_shape(self, kernel_shape):

        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise ValueError(
                "Point spread function kernel shape must be odd in both dimensions, but shape "
                + str(kernel_shape)
                + " was provided."
            )
        else:
            self._point_spread_kernel_shape = kernel_shape

    def render(
        self,
        peaks_dict,
        frames_to_render=0,
        powder=False,
        lorentz=True,
        polarization=True,
        structure_factor=True,
        method="centroid",
    ):

        # Intersect scattering vectors with detector plane
        zd_yd_angle = self.get_intersection(
            peaks_dict["peaks"][:, 13:16], peaks_dict["peaks"][:, 16:19]
        )

        peaks_dict["peaks"] = torch.cat((peaks_dict["peaks"], zd_yd_angle), dim=1)
        peaks_dict["columns"].extend(["zd", "yd", "incident_angle"])

        """
        Column names of peaks are
        0: 'grain_index'        10: 'Gx'        20: 'polarization_factors'
        1: 'phase_number'       11: 'Gy'        21: 'volumes'
        2: 'h'                  12: 'Gz'        22: 'zd'
        3: 'k'                  13: 'K_out_x'   23: 'yd'
        4: 'l'                  14: 'K_out_y'   24: 'incident_angle'
        5: 'structure_factors'  15: 'K_out_z'
        6: 'diffraction_times'  16: 'Source_x'
        7: 'G0_x'              17: 'Source_y'
        8: 'G0_y'              18: 'Source_z'
        9: 'G0_z'              19: 'lorentz_factors'
        """

        if powder is True:
            diffraction_frames = self.project_peaks(
                peaks_dict, frames_to_render, lorentz, polarization, structure_factor
            )
        else:
            diffraction_frames = self.project_scattering_units(
                peaks_dict,
                frames_to_render,
                lorentz,
                polarization,
                structure_factor,
                method,
            )

        return diffraction_frames

    def project_peaks(
        self, peaks_dict, frames_to_render, lorentz, polarization, structure_factor
    ):

        peaks = peaks_dict["peaks"]

        # Filter out peaks not hitting the detector
        peaks = peaks[self.contains(peaks[:, 22], peaks[:, 23])]

        # Add frame number at the end of the tensor
        bin_edges = torch.linspace(0, 1, steps=frames_to_render)
        frame = torch.bucketize(peaks[:, 6].contiguous(), bin_edges).unsqueeze(1) - 1
        peaks = torch.cat((peaks, frame), dim=1)
        peaks_dict["columns"].append("frame")

        """
        Column names of peaks are
        0: 'grain_index'        10: 'Gx'        20: 'polarization_factors'
        1: 'phase_number'       11: 'Gy'        21: 'volumes'
        2: 'h'                  12: 'Gz'        22: 'zd'
        3: 'k'                  13: 'K_out_x'   23: 'yd'
        4: 'l'                  14: 'K_out_y'   24: 'incident_angle'
        5: 'structure_factors'  15: 'K_out_z'   25: 'frame'
        6: 'diffraction_times'  16: 'Source_x'
        7: 'G0_x'              17: 'Source_y'
        8: 'G0_y'              18: 'Source_z'
        9: 'G0_z'              19: 'lorentz_factors'
        """

        # Create a 3 colum matrix with X,Y and frame coordinates for each peak
        pixel_indices = torch.cat(
            (
                ((peaks[:, 22]) / self.pixel_size_z).unsqueeze(1),
                ((peaks[:, 23]) / self.pixel_size_y).unsqueeze(1),
                peaks[:, 25].unsqueeze(1),
            ),
            dim=1,
        ).to(torch.int32)

        """This truncates the pixels to zero, which shifts the position of dose deposited by 1 pixel
        in x and 1 pixel in y compared to previous versions of the code which was rounding up.
        """

        frames_n = peaks[:, 25].unique().shape[0]

        # Create the future frames as an empty tensor
        diffraction_frames = torch.zeros(
            (frames_n, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1])
        )
        # Generate the relative intensity for all the diffraction peaks using the different factors.
        structure_factors = peaks[:, 5]
        lorentz_factors = peaks[:, 19]
        polarization_factors = peaks[:, 20]
        volumes = peaks[:, 21]

        relative_intensity = (
            (structure_factors * volumes)  # small numbers first
            * polarization_factors  # medium range
            * lorentz_factors  # large numbers last
        )

        # Turn from lists of peaks to rendered frames
        # Step 1: Find unique coordinates and the inverse indices
        unique_coords, inverse_indices = torch.unique(
            pixel_indices, dim=0, return_inverse=True
        )

        # Step 2: Count occurrences of each unique coordinate, weighting by the relative intensity
        counts = torch.bincount(inverse_indices, weights=relative_intensity)

        # Step 3: Combine unique coordinates and their counts into a new tensor (mx4)
        result = torch.cat((unique_coords, counts.unsqueeze(1)), dim=1).type_as(
            diffraction_frames
        )

        # Step 4: Use the new column as a pixel value to be added to each coordinate
        diffraction_frames[
            result[:, 2].int(), result[:, 0].int(), result[:, 1].int()
        ] = result[:, 3]

        diffraction_frames = self._apply_point_spread_function(diffraction_frames)

        return diffraction_frames

    def project_scattering_units(
        self,
        peaks_dict,
        frames_to_render=1,
        lorentz=True,
        polarization=True,
        structure_factor=True,
        method="centroid",
    ):
        """Render a pixelated diffraction frame onto the detector plane .

        NOTE: The value read out on a pixel in the detector is an approximation of the integrated number of counts over
        the pixel area.

        Args:
            frames_to_render (:obj:`int` or :obj:`iterable` of :obj:`int` or :obj:`str`): Indices of the frame in the :obj:`frames` list
                to be rendered. Optionally the keyword string 'all' can be passed to render all frames of the detector.
            lorentz (:obj:`bool`): Weight scattered intensity by Lorentz factor. Defaults to False.
            polarization (:obj:`bool`): Weight scattered intensity by Polarization factor. Defaults to False.
            structure_factor (:obj:`bool`): Weight scattered intensity by Structure Factor factor. Defaults to False.
            method (:obj:`str`): Rendering method, must be one of ```project``` , ```centroid``` or ```centroid_with_scintillator```.
                Defaults to ```centroid```. The default,```method=centroid```, is a simple deposit of intensity for each scattering_unit
                onto the detector by tracing a line from the sample scattering region centroid to the detector plane. The intensity
                is deposited into a single detector pixel regardless of the geometrical shape of the scattering_unit. If instead
                ```method=project``` the scattering regions are projected onto the detector depositing a intensity over
                possibly several pixels as weighted by the optical path lengths of the rays diffracting from the scattering
                region. If ```method=centroid_with_scintillator`` a centroid type raytracing is used, but the scintillator
                point spread is applied before deposition unto the pixel grid, revealing sub pixel shifts in the scattering events.
            verbose (:obj:`bool`): Prints progress. Defaults to True.
            number_of_processes (:obj:`int`): Optional keyword specifying the number of desired processes to use for diffraction
                computation. Defaults to 1, i.e a single processes.
        Returns:
            A pixelated frame as a (:obj:`numpy array`) with shape inferred form the detector geometry and
            pixel size.

        NOTE: This function can be overwitten to do more advanced models for intensity.

        """

        if method == "project":
            renderer = self._projection_render
            kernel = self._get_point_spread_function_kernel()
        elif method == "centroid":
            renderer = self._centroid_render
            kernel = self._get_point_spread_function_kernel()
        elif method == "centroid_with_scintillator":
            renderer = self._centroid_render_with_scintillator
            kernel = None
        else:
            raise ValueError(
                "No such method: "
                + method
                + " exist, method should be one of project or centroid"
            )

        frames_bundle = list(range(frames_to_render))

        diffraction_frames = self._render_and_convolve(
            peaks_dict,
            frames_bundle,
            kernel,
            renderer,
            lorentz,
            polarization,
            structure_factor,
        )

        return diffraction_frames.squeeze()

    def _render_and_convolve(
        self,
        peaks_dict,
        frames_bundle,
        kernel,
        renderer,
        lorentz,
        polarization,
        structure_factor,
        verbose=True,
    ):
        scattering_units = peaks_dict["scattering_units"]
        diffraction_frames = []
        for frame_index in frames_bundle:
            frame = np.zeros(
                (self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1])
            )
            for i, scattering_unit in enumerate(scattering_units):
                zd = peaks_dict["peaks"][i, 22]
                yd = peaks_dict["peaks"][i, 23]
                renderer(
                    scattering_unit,
                    zd,
                    yd,
                    frame,
                    lorentz,
                    polarization,
                    structure_factor,
                )
            if kernel is not None:
                frame = self._apply_point_spread_function(frame, kernel)
            diffraction_frames.append(frame)
        return np.array(diffraction_frames)

    def _apply_point_spread_function(self, frames):
        """Apply the point spread function to the detector frames."""
        kernel = ensure_torch(self._get_point_spread_function_kernel())

        if frames.ndim == 2:
            frames = frames.unsqueeze(0)  # Add channel dimension if only 1 image
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)  # Add channel dimension if only 1 image
        frames_n = frames.shape[1]

        kernel = kernel.repeat(frames_n, frames_n, 1, 1)

        # Perform the convolution
        with torch.no_grad():
            output = torch.nn.functional.conv2d(frames, weight=kernel, padding=1)

        return output

    def pixel_index_to_theta_eta(
        self,
        incoming_wavevector,
        pixel_zd_index,
        pixel_yd_index,
        scattering_origin=ensure_torch([0, 0, 0]),
    ):
        """Compute bragg angle and azimuth angle for a detector pixel index.

        Args:
            pixel_zd_index (:obj:`float`): Coordinate in microns along detector zd axis.
            pixel_yd_index (:obj:`float`): Coordinate in microns along detector yd axis.
            scattering_origin (obj:`numpy array`): Origin of diffraction in microns. Defaults to ensure_torch([0, 0, 0]).

        Returns:
            (:obj:`tuple`) Bragg angle theta and azimuth angle eta (measured from det_corner_1 - det_corner_0 axis) in radians
        """
        # TODO: unit test
        pixel_zd_coord = pixel_zd_index * self.pixel_size_z
        pixel_yd_coord = pixel_yd_index * self.pixel_size_y
        theta, eta = self.pixel_coord_to_theta_eta(
            incoming_wavevector,
            pixel_zd_coord,
            pixel_yd_coord,
            scattering_origin=scattering_origin,
        )
        return theta, eta

    def pixel_coord_to_theta_eta(
        self,
        incoming_wavevector,
        pixel_zd_coord,
        pixel_yd_coord,
        scattering_origin=ensure_torch([0, 0, 0]),
    ):
        """Compute bragg angle and azimuth angle  for a detector coordinate.

        Args:
            pixel_zd_coord (:obj:`float`): Coordinate in microns along detector zd axis.
            pixel_yd_coord (:obj:`float`): Coordinate in microns along detector yd axis.
            scattering_origin (obj:`numpy array`): Origin of diffraction in microns. Defaults to ensure_torch([0, 0, 0]).

        Returns:
            (:obj:`tuple`) Bragg angle theta and azimuth angle eta (measured from det_corner_1 - det_corner_0 axis) in radians
        """
        # TODO: unit test
        khat = incoming_wavevector / np.linalg.norm(incoming_wavevector)
        kp = (
            self.det_corner_0
            + pixel_zd_coord * self.zdhat
            + pixel_yd_coord * self.ydhat
            - scattering_origin
        )
        kprimehat = kp / np.linalg.norm(kp)
        theta = np.arccos(khat.dot(kprimehat)) / 2.0
        korthogonal = kprimehat - (khat * kprimehat.dot(khat))
        eta = np.arccos(self.zdhat.dot(korthogonal) / np.linalg.norm(korthogonal))
        eta *= np.sign((np.cross(self.zdhat, korthogonal)).dot(-incoming_wavevector))
        return theta, eta

    def get_intersection(self, ray_direction, source_point):
        """Get detector intersection in detector coordinates of every single ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vectors in direction of the xray propagation
            source_point (:obj:`numpy array`): Origin of the ray.

        Returns:
            (:obj:`tuple`) zd, yd in detector plane coordinates.

        """
        s = torch.matmul(self.det_corner_0 - source_point, self.normal) / torch.matmul(
            ray_direction, self.normal
        )

        intersection = source_point + ray_direction * s.unsqueeze(1)
        # such that backwards rays are not considered to intersect the detector
        # i.e only rays that can intersect the detector plane by propagating
        # forward along the photon path are considered.
        intersection[s < 0] = np.nan
        zd = torch.matmul(intersection - self.det_corner_0, self.zdhat)
        yd = torch.matmul(intersection - self.det_corner_0, self.ydhat)

        # Calculate incident angle
        ray_dir_norm = ray_direction / torch.norm(ray_direction, dim=1).unsqueeze(-1)
        normal_norm = self.normal / torch.linalg.norm(self.normal)
        cosine_theta = torch.matmul(
            ray_dir_norm, -normal_norm
        )  # The detector normal by default goes against the beam
        incident_angle_deg = torch.arccos(cosine_theta) * (180 / torch.pi)
        return torch.stack((zd, yd, incident_angle_deg), dim=1)

    def contains(self, zd, yd):
        """Determine if the detector coordinate zd,yd lies within the detector bounds.

        Args:
            zd (:obj:`float`): Detector z coordinate
            yd (:obj:`float`): Detector y coordinate

        Returns:
            (:obj:`boolean`) True if the zd,yd is within the detector bounds.

        """
        return (zd >= 0) & (zd <= self.zmax) & (yd >= 0) & (yd <= self.ymax)

    def project(self, scattering_unit, box):
        """Compute parametric projection of scattering region unto detector.

        Args:
            scattering_unit (:obj:`xrd_simulator.ScatteringUnit`): The scattering region.
            box (:obj:`tuple` of :obj:`int`): indices of the detector frame over which to compute the projection.
                i.e the subgrid of the detector is taken as: array[[box[0]:box[1], box[2]:box[3]].

        Returns:
            (:obj:`numpy array`) clip lengths between scattering_unit polyhedron and rays traced from the detector.

        """

        ray_points = self.pixel_coordinates[
            box[0] : box[1], box[2] : box[3], :
        ].reshape((box[1] - box[0]) * (box[3] - box[2]), 3)

        plane_normals = scattering_unit.convex_hull.equations[:, 0:3]
        plane_ofsets = scattering_unit.convex_hull.equations[:, 3].reshape(
            scattering_unit.convex_hull.equations.shape[0], 1
        )
        plane_points = -np.multiply(plane_ofsets, plane_normals)

        ray_points = np.ascontiguousarray(ray_points)
        ray_direction = np.ascontiguousarray(
            scattering_unit.scattered_wave_vector
            / np.linalg.norm(scattering_unit.scattered_wave_vector)
        )
        plane_points = np.ascontiguousarray(plane_points)
        plane_normals = np.ascontiguousarray(plane_normals)

        clip_lengths = utils._clip_line_with_convex_polyhedron(
            ray_points, ray_direction, plane_points, plane_normals
        )
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
        fourth_corner_of_detector = self.det_corner_2 + (
            self.det_corner_1 - self.det_corner_0[:]
        )
        geom_mat = torch.zeros((3, 4))
        for i, det_corner in enumerate(
            [
                self.det_corner_0,
                self.det_corner_1,
                self.det_corner_2,
                fourth_corner_of_detector,
            ]
        ):
            geom_mat[:, i] = det_corner - source_point
        normalised_local_coord_geom_mat = geom_mat / torch.linalg.norm(geom_mat, axis=0)
        cone_opening = torch.arccos(
            torch.matmul(normalised_local_coord_geom_mat.T, k / torch.linalg.norm(k))
        )  # These are two time Bragg angles
        return torch.max(cone_opening) / 2.0

    def save(self, path):
        """Save the detector object to disc (via pickling). Change the arrays formats to np first.

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
        with open(path, "rb") as f:
            loaded = dill.load(f)
            loaded.normal = ensure_torch(loaded.normal)
            loaded.det_corner_0 = ensure_torch(loaded.det_corner_0)
            loaded.det_corner_1 = ensure_torch(loaded.det_corner_1)
            loaded.det_corner_2 = ensure_torch(loaded.det_corner_2)
            loaded.zdhat = ensure_torch(loaded.zdhat)
            loaded.ydhat = ensure_torch(loaded.ydhat)
            loaded.zmax = ensure_torch(loaded.zmax)
            loaded.ymax = ensure_torch(loaded.ymax)
            loaded.pixel_size_z = ensure_torch(loaded.pixel_size_z)
            loaded.pixel_size_y = ensure_torch(loaded.pixel_size_y)
            return loaded

    def _get_point_spread_function_kernel(self):
        """Render the point_spread_function onto a grid of shape specified by point_spread_kernel_shape."""
        sz, sy = self.point_spread_kernel_shape
        axz = np.linspace(-(sz - 1) / 2.0, (sz - 1) / 2.0, sz)
        axy = np.linspace(-(sy - 1) / 2.0, (sy - 1) / 2.0, sy)
        Z, Y = np.meshgrid(axz, axy, indexing="ij")
        kernel = np.zeros(self.point_spread_kernel_shape)
        for i in range(Z.shape[0]):
            for j in range(Y.shape[1]):
                kernel[i, j] = self.point_spread_function(Z[i, j], Y[i, j])
        assert (
            len(kernel[kernel < 0]) == 0
        ), "Point spread function must be strictly positive, but negative values were found."
        assert (
            np.sum(kernel) > 1e-8
        ), "The integrated value of the point spread function over the defined kernel domain is close to zero."

        return kernel / np.sum(kernel)

    def _get_pixel_coordinates(self):
        zds = torch.arange(0, self.zmax, self.pixel_size_z)
        yds = torch.arange(0, self.ymax, self.pixel_size_y)
        Z, Y = torch.meshgrid(zds, yds, indexing="ij")
        Zds = torch.zeros((len(zds), len(yds), 3))
        Yds = torch.zeros((len(zds), len(yds), 3))
        for i in range(3):
            Zds[:, :, i] = Z
            Yds[:, :, i] = Y
        pixel_coordinates = (
            self.det_corner_0.reshape(1, 1, 3)
            + Zds * self.zdhat.reshape(1, 1, 3)
            + Yds * self.ydhat.reshape(1, 1, 3)
        )
        return pixel_coordinates

    def _centroid_render(
        self, scattering_unit, zd, yd, frame, lorentz, polarization, structure_factor
    ):
        """Simple deposit of intensity for each scattering_unit onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited into a single
        detector pixel regardless of the geometrical shape of the scattering_unit.
        """
        # zd, yd = scattering_unit.zd, scattering_unit.yd

        if self.contains(zd, yd):
            intensity_scaling_factor = self._get_intensity_factor(
                scattering_unit, lorentz, polarization, structure_factor
            )
            row, col = self._detector_coordinate_to_pixel_index(zd, yd)
            if np.isinf(intensity_scaling_factor):
                frame[row, col] += np.inf
            else:
                frame[row, col] += scattering_unit.volume * intensity_scaling_factor

    def _centroid_render_with_scintillator(
        self, scattering_unit, zd, yd, frame, lorentz, polarization, structure_factor
    ):
        """Simple deposit of intensity for each scattering_unit onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited by placing the detector
        point spread function at the hit location and rendering it unto the detector grid.

        NOTE: this is different from self._centroid_render which applies the point spread function as a post-proccessing
        step using convolution. Here the point spread is simulated to take place in the scintillator, before reaching the
        chip.
        """
        # zd, yd = scattering_unit.zd, scattering_unit.yd
        if self.contains(zd, yd):
            intensity_scaling_factor = self._get_intensity_factor(
                scattering_unit, lorentz, polarization, structure_factor
            )

            a, b = self.point_spread_kernel_shape
            row, col = self._detector_coordinate_to_pixel_index(zd, yd)
            zd_in_pixels = zd / self.pixel_size_z
            yd_in_pixels = yd / self.pixel_size_y
            rl, rh = row - a // 2 - 1, row + a // 2 + 1
            cl, ch = col - b // 2 - 1, col + b // 2 + 1
            rl, rh = np.max([rl, 0]), np.min([rh, self.pixel_coordinates.shape[0] - 1])
            cl, ch = np.max([cl, 0]), np.min([ch, self.pixel_coordinates.shape[1] - 1])
            zg = np.linspace(rl, rh, rh - rl + 1) + 0.5  # pixel centre coordinates in z
            yg = np.linspace(cl, ch, ch - cl + 1) + 0.5  # pixel centre coordinates in y
            Z, Y = np.meshgrid(zg, yg, indexing="ij")
            drifted_kernel = self.point_spread_function(
                Z - zd_in_pixels, Y - yd_in_pixels
            )
            drifted_kernel = drifted_kernel / np.sum(drifted_kernel)

            if np.isinf(intensity_scaling_factor):
                frame[rl : rh + 1, cl : ch + 1] = np.inf
            else:
                frame[rl : rh + 1, cl : ch + 1] += (
                    scattering_unit.volume * intensity_scaling_factor * drifted_kernel
                )

    def _projection_render(
        self, scattering_unit, frame, lorentz, polarization, structure_factor
    ):
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
                    scattering_unit, frame, lorentz, polarization, structure_factor
                )
            else:
                intensity_scaling_factor = self._get_intensity_factor(
                    scattering_unit, lorentz, polarization, structure_factor
                )
                if np.isinf(intensity_scaling_factor):
                    frame[box[0] : box[1], box[2] : box[3]] += np.inf
                else:
                    frame[box[0] : box[1], box[2] : box[3]] += (
                        projection
                        * intensity_scaling_factor
                        * self.pixel_size_z
                        * self.pixel_size_y
                    )

    def _get_intensity_factor(
        self, scattering_unit, lorentz, polarization, structure_factor
    ):
        intensity_factor = 1.0
        # TODO: Consider solid angle intensity rescaling and air scattering.
        if lorentz:
            intensity_factor *= scattering_unit.lorentz_factor
        if polarization:
            intensity_factor *= scattering_unit.polarization_factor
        if structure_factor:
            if scattering_unit.phase.structure_factors is not None:
                intensity_factor *= (
                    scattering_unit.real_structure_factor**2
                    + scattering_unit.imaginary_structure_factor**2
                )
            else:
                raise ValueError(
                    "Structure factors have not been set, .cif file is required at sample instantiation."
                )

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
        vertices = scattering_unit.convex_hull.points[
            scattering_unit.convex_hull.vertices
        ]

        projected_vertices = self.get_intersection(
            scattering_unit.scattered_wave_vector, vertices
        )

        min_zd, max_zd = np.min(projected_vertices[:, 0]), np.max(
            projected_vertices[:, 0]
        )
        min_yd, max_yd = np.min(projected_vertices[:, 1]), np.max(
            projected_vertices[:, 1]
        )

        min_zd, max_zd = np.max([min_zd, 0]), np.min([max_zd, self.zmax])
        min_yd, max_yd = np.max([min_yd, 0]), np.min([max_yd, self.ymax])

        if min_zd > max_zd or min_yd > max_yd:
            return None

        min_row_indx, min_col_indx = self._detector_coordinate_to_pixel_index(
            min_zd, min_yd
        )
        max_row_indx, max_col_indx = self._detector_coordinate_to_pixel_index(
            max_zd, max_yd
        )

        max_row_indx = np.min([max_row_indx + 1, int(self.zmax / self.pixel_size_z)])
        max_col_indx = np.min([max_col_indx + 1, int(self.ymax / self.pixel_size_y)])

        return min_row_indx, max_row_indx, min_col_indx, max_col_indx

    def to_torch(self):
        self.det_corner_0 = ensure_torch(self.det_corner_0)
        self.det_corner_1 = ensure_torch(self.det_corner_1)
        self.det_corner_2 = ensure_torch(self.det_corner_2)

        self.pixel_size_z = ensure_torch(self.pixel_size_z)
        self.pixel_size_y = ensure_torch(self.pixel_size_y)

        self.zmax = ensure_torch(self.zmax)
        self.ymax = ensure_torch(self.ymax)

        self.zdhat = ensure_torch(self.zdhat)
        self.ydhat = ensure_torch(self.ydhat)
        self.normal = ensure_torch(self.normal)
        self.pixel_coordinates = ensure_torch(self.pixel_coordinates)
