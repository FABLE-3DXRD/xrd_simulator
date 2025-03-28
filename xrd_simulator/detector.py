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
from xrd_simulator.utils import ensure_torch, peaks_to_csv
import dill
import torch
import torch.nn.functional as F
from scipy.special import wofz

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
        gaussian_sigma (:obj:`float`, optional): Standard deviation of the Gaussian point spread function in pixels.
            Defaults to 1.0.
        kernel_threshold (:obj:`float`, optional): Intensity threshold used to determine the kernel size.
            The kernel will extend until the Gaussian tail drops below this value. Defaults to 0.05.

    Attributes:
        pixel_size_z (:obj:`float`): Pixel side length along zdhat (rectangular pixels) in units of microns.
        pixel_size_y (:obj:`float`): Pixel side length along ydhat (rectangular pixels) in units of microns.
        det_corner_0,det_corner_1,det_corner_2 (:obj:`numpy array`): Detector corner 3d coordinates ``shape=(3,)``.
            The origin of the detector is at det_corner_0.
        frames (:obj:`list` of :obj:`list` of :obj:`scattering_unit.ScatteringUnit`): Analytical diffraction frames
        zdhat,ydhat (:obj:`numpy array`): Detector basis vectors.
        normal (:obj:`numpy array`): Detector normal, formed as the cross product: numpy.cross(self.zdhat, self.ydhat)
        zmax,ymax (:obj:`numpy array`): Detector width and height.
        pixel_coordinates  (:obj:`numpy array`): Real space 3d detector pixel coordinates. ``shape=(n,3)``
        gaussian_sigma (:obj:`float`): Standard deviation of the Gaussian point spread function.
        kernel_threshold (:obj:`float`): Threshold value that determines the kernel size.
        gaussian_kernel (:obj:`torch.Tensor`): Pre-computed normalized 2D Gaussian kernel.

    """

    def __init__(
        self,
        pixel_size_z,
        pixel_size_y,
        det_corner_0,
        det_corner_1,
        det_corner_2,
        gaussian_sigma=1.0,
        kernel_threshold=0.05,
        use_lorentz=True,
        use_polarization=True,
        use_structure_factor=True,
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
        self.gaussian_sigma = gaussian_sigma
        self.kernel_threshold = kernel_threshold
        self.gaussian_kernel = self.gaussian_kernel_2d()

        self.lorentz_factor = use_lorentz
        self.polarization_factor = use_polarization
        self.structure_factor = use_structure_factor

    def render(
        self,
        peaks_dict,
        frames_to_render=0,
        method="gauss",
    ):
        peaks_dict = self._peaks_detector_intersection(peaks_dict, frames_to_render)

        """
            Column names of peaks are
            0: 'grain_index'        10: 'Gx'        20: 'polarization_factors'
            1: 'phase_number'       11: 'Gy'        21: 'volumes'
            2: 'h'                  12: 'Gz'        22: '2theta'
            3: 'k'                  13: 'K_out_x'   23: 'scherrer_fwhm'
            4: 'l'                  14: 'K_out_y'   24: 'zd'
            5: 'structure_factors'  15: 'K_out_z'   25: 'yd'
            6: 'diffraction_times'  16: 'Source_x'  26: 'incident_angle'
            7: 'G0_x'               17: 'Source_y'  27: 'frame'
            8: 'G0_y'               18: 'Source_z'  28: 'relative_intensity'
            9: 'G0_z'               19: 'lorentz_factors'           
        """
        if method == "gauss":
            diffraction_frames = self._render_gauss_peaks(peaks_dict["peaks"])
        elif method == "voigt":
            diffraction_frames = self._render_voigt_peaks(peaks_dict["peaks"])
        elif method == "volumes":
            diffraction_frames = self._render_projected_volumes(peaks_dict)
        else:
            raise ValueError(
                f"Invalid method: {method}. Must be one of: 'gauss', 'voigt', or 'volumes'"
            )

        return diffraction_frames

    def _render_gauss_peaks(
        self,
        peaks,
    ):

        # Create a 3 colum matrix with X,Y and frame coordinates for each peak

        pixel_indices = torch.cat(
            (
                ((peaks[:, 24]) / self.pixel_size_z).unsqueeze(1),
                ((peaks[:, 25]) / self.pixel_size_y).unsqueeze(1),
                peaks[:, 27].unsqueeze(1),
            ),
            dim=1,
        ).to(torch.int32)

        frames_n = peaks[:, 27].unique().shape[0]

        # Create the future frames as an empty tensor
        diffraction_frames = torch.zeros(
            (
                frames_n,
                self.pixel_coordinates.shape[0],
                self.pixel_coordinates.shape[1],
            )
        )

        # Turn from lists of peaks to rendered frames
        # Step 1: Find unique coordinates and the inverse indices
        unique_coords, inverse_indices = torch.unique(
            pixel_indices, dim=0, return_inverse=True
        )

        # Step 2: Count occurrences of each unique coordinate, weighting by the relative intensity
        counts = torch.bincount(inverse_indices, weights=peaks[:, 28])

        # Step 3: Combine unique coordinates and their counts into a new tensor (mx4)
        result = torch.cat((unique_coords, counts.unsqueeze(1)), dim=1).type_as(
            diffraction_frames
        )

        # Step 4: Use the new column as a pixel value to be added to each coordinate
        diffraction_frames[
            result[:, 2].int(), result[:, 0].int(), result[:, 1].int()
        ] = result[:, 3]

        diffraction_frames = self._conv2d_gaussian_kernel(diffraction_frames)

        return diffraction_frames

    def _render_voigt_peaks(self, peaks: torch.Tensor) -> torch.Tensor:
        """Deposit Voigt kernels starting with small batches and gradually increasing.

        Parameters
        ----------
        peaks : torch.Tensor
            Processed peaks tensor with precalculated intensities and frame indices
        initial_batch_size : int, optional
            Initial number of peaks to process per batch, by default 10
        max_batch_size : int, optional
            Maximum batch size to try, by default 1000

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames
        """
        crystallite_size = (
            2.0
            * (3 * peaks[:, 21] / (4 * np.pi)) ** (1 / 3)
            * 1000  # microns to nanometers
        )
        q25, q50, q75 = torch.quantile(
            crystallite_size, torch.tensor([0.25, 0.5, 0.75])
        )
        print(
            f"\nCrystallite size quartiles (nm):"
            f"\n  25th: {q25:.2f}"
            f"\n  50th: {q50:.2f}"
            f"\n  75th: {q75:.2f}"
            f"\nCrystallite size range: {crystallite_size.min():.2f} - {crystallite_size.max():.2f}"
        )
        batch_size = int(crystallite_size.min() * 10)
        frames_n = int(peaks[:, 27].max() + 1)
        frames = torch.zeros(
            (frames_n, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1])
        )

        for frame_idx in range(frames_n):
            frame_mask = peaks[:, 27] == frame_idx
            frame_peaks = peaks[frame_mask]

            if len(frame_peaks) == 0:
                continue

            print(
                f"Processing frame {frame_idx+1}/{frames_n} with {len(frame_peaks)} peaks"
            )

            start_idx = 0

            while start_idx < len(frame_peaks):

                # Process current batch
                end_idx = min(start_idx + batch_size, len(frame_peaks))
                batch_peaks = frame_peaks[start_idx:end_idx]

                # Get parameters for this batch
                fwhm_rad = batch_peaks[:, 23]
                incident_angles = batch_peaks[:, 26]
                zd = batch_peaks[:, 24] / self.pixel_size_z
                yd = batch_peaks[:, 25] / self.pixel_size_y
                intensities = batch_peaks[:, 28]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Generate and deposit kernels
                kernels = self._voigt_kernel_batch(fwhm_rad, incident_angles)
                kernels = kernels * intensities.view(-1, 1, 1, 1)
                frames[frame_idx] = self._deposit_kernels_batch(
                    frames[frame_idx], kernels, zd, yd
                )

                # Success - move to next batch
                del kernels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                start_idx = end_idx

                # Report progress
                progress = int((start_idx / len(frame_peaks)) * 100)
                print(
                    f"Frame {frame_idx+1}: {progress}% complete (batch: {batch_size})"
                )

            print(f"Completed frame {frame_idx+1}/{frames_n}")

        return frames

    def _render_projected_volumes(
        self,
        peaks_dict,
        renderer,
    ):
        scattering_units = peaks_dict["scattering_units"]
        kernel = self.gaussian_kernel
        frames_bundle = peaks_dict["peaks"][:, 27].unique()
        diffraction_frames = []
        for frame_index in frames_bundle:
            frames = np.zeros(
                (self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1])
            )
            for i, scattering_unit in enumerate(scattering_units):
                zd = peaks_dict["peaks"][i, 22]
                yd = peaks_dict["peaks"][i, 23]
                renderer(
                    scattering_unit,
                    zd,
                    yd,
                    frames,
                    self.lorentz_factor,
                    self.polarization_factor,
                    self.structure_factor,
                )
            if kernel is not None:
                frames = self._conv2d_gaussian_kernel(frames)
            diffraction_frames.append(frames)
        return np.array(diffraction_frames)

    def _conv2d_gaussian_kernel(self, frames):
        """Apply the point spread function to the detector frames."""
        # kernel = ensure_torch(self._get_point_spread_function_kernel())
        frames = ensure_torch(frames)
        if frames.ndim == 2:
            frames = frames.unsqueeze(0)  # Add channel dimension if only 1 image
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)  # Add channel dimension if only 1 image

        # kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = self.gaussian_kernel
        padding = kernel.shape[-1] // 2
        # Perform the convolution
        with torch.no_grad():
            output = torch.nn.functional.conv2d(frames, weight=kernel, padding=padding)

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
            pixel_zd_index (:obj=`float`): Coordinate in microns along detector zd axis.
            pixel_yd_index (:obj=`float`): Coordinate in microns along detector yd axis.
            scattering_origin (obj:`numpy array`): Origin of diffraction in microns. Defaults to ensure_torch([0, 0, 0]).

        Returns:
            (:obj=`tuple`) Bragg angle theta and azimuth angle eta (measured from det_corner_1 - det_corner_0 axis) in radians
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
            pixel_zd_coord (:obj=`float`): Coordinate in microns along detector zd axis.
            pixel_yd_coord (:obj=`float`): Coordinate in microns along detector yd axis.
            scattering_origin (obj:`numpy array`): Origin of diffraction in microns. Defaults to ensure_torch([0, 0, 0]).

        Returns:
            (:obj=`tuple`) Bragg angle theta and azimuth angle eta (measured from det_corner_1 - det_corner_0 axis) in radians
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
            ray_direction (:obj=`numpy array`): Vectors in direction of the xray propagation
            source_point (:obj=`numpy array`): Origin of the ray.

        Returns:
            (:obj=`tuple`) zd, yd in detector plane coordinates.

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
            zd (:obj=`float`): Detector z coordinate
            yd (:obj=`float`): Detector y coordinate

        Returns:
            (:obj=`boolean`) True if the zd,yd is within the detector bounds.

        """
        return (zd >= 0) & (zd <= self.zmax) & (yd >= 0) & (yd <= self.ymax)

    def project_convex_hull(self, scattering_unit, box):
        """Compute parametric projection of scattering region unto detector.

        Args:
            scattering_unit (:obj=`xrd_simulator.ScatteringUnit`): The scattering region.
            box (:obj=`tuple` of :obj=`int`): indices of the detector frame over which to compute the projection.
                i.e the subgrid of the detector is taken as: array[[box[0]:box[1], box[2]:box[3]].

        Returns:
            (:obj=`numpy array`) clip lengths between scattering_unit polyhedron and rays traced from the detector.

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
            k (:obj=`numpy array`): Wavevector forming the central axis of cone ```shape=(3,)```.
            source_point (:obj=`numpy array`): Origin of the wavevector ```shape=(3,)```.

        Returns:
            (:obj=`float`) Cone opening angle divided by two (radians), corresponding to a maximum bragg angle after
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
            path (:obj=`str`): File path at which to save, ending with the desired filename.

        """

        if not path.endswith(".det"):
            path = path + ".det"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the detector object from disc (via pickling).

        Args:
            path (:obj=`str`): File path at which to load, ending with the desired filename.

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

    def gaussian_kernel_2d(self) -> torch.Tensor:
        """
        Generates a normalized 2D Gaussian kernel with dynamic size based on intensity threshold.

        Args:
            sigma (float): Standard deviation of the Gaussian.
            threshold (float): Allowed remaining intensity outside the kernel (default 5%).

        Returns:
            torch.Tensor: 2D Gaussian kernel of shape (1, 1, H, W).
        """
        sigma = ensure_torch(self.gaussian_sigma)
        threshold = self.kernel_threshold

        # Determine the radius where tail drops below threshold / 2 (1D axis)
        radius = 1
        while True:
            value = torch.exp(-(radius**2) / (2 * sigma**2))
            if value < threshold / 2:
                break
            radius += 1

        kernel_size = 2 * radius + 1  # Ensure odd size

        # Create coordinate grid
        ax = torch.arange(kernel_size, dtype=torch.float64) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel.view(1, 1, kernel_size, kernel_size)

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

    def _projection_render(
        self, scattering_unit, frame, lorentz, polarization, structure_factor
    ):
        """Raytrace and project the scattering regions onto the detector plane for increased peak shape accuracy.
        This is generally very computationally expensive compared to the simpler (:obj=`function`):_centroid_render
        function.

        NOTE: If the projection of the scattering_unit does not hit any pixel centroids of the detector fallback to
        the (:obj=`function`):_centroid_render function is used to deposit the intensity into a single detector pixel.
        """
        box = self._get_projected_bounding_box(scattering_unit)
        if box is not None:
            projection = self.project_convex_hull(scattering_unit, box)
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
            scattering_unit (:obj=`xrd_simulator.ScatteringUnit`): The scattering region.

        Returns:
            (:obj=`tuple` of :obj=`int`) indices that can be used to slice the detector frame array and get the pixels that
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

    def _peaks_detector_intersection(self, peaks_dict, frames_to_render):
        """
        Computes detector intersections, filters visible peaks, and assigns frame indices.
        Also calculates and stores combined intensity factors.

        Parameters
        ----------
        peaks_dict : dict
            Dictionary containing peaks information
        frames_to_render : int
            Number of frames to render

        Returns
        -------
        dict
            Updated peaks dictionary with intersections and intensity information
        """
        peaks = peaks_dict["peaks"]

        # Intersect scattering vectors with detector plane
        zd_yd_angle = self.get_intersection(peaks[:, 13:16], peaks[:, 16:19])
        peaks = torch.cat((peaks, zd_yd_angle), dim=1)
        peaks_dict["columns"].extend(["zd", "yd", "incident_angle"])

        # Filter visible peaks
        mask = self.contains(peaks[:, 24], peaks[:, 25])
        peaks = peaks[mask]

        # Assign frame index based on normalized diffraction time (column 6)
        time = peaks[:, 6].contiguous()
        bins = torch.linspace(0, 1, frames_to_render).contiguous()
        frame = torch.bucketize(time, bins).unsqueeze(1) - 1
        peaks = torch.cat((peaks, frame), dim=1)
        peaks_dict["columns"].append("frame")

        # Calculate combined intensity factors
        intensity = peaks[:, 21]  # volumes
        if self.structure_factor:
            intensity = intensity * peaks[:, 5]  # structure_factors
        if self.polarization_factor:
            intensity = intensity * peaks[:, 20]  # polarization_factors
        if self.lorentz_factor:
            intensity = intensity * peaks[:, 19]  # lorentz_factors

        # Add intensity as new column
        intensity = intensity.unsqueeze(1)
        peaks = torch.cat((peaks, intensity), dim=1)
        peaks_dict["columns"].append("intensity")

        peaks_dict["peaks"] = peaks
        return peaks_dict

    def _voigt_kernel_batch(
        self, fwhm_rad: torch.Tensor, incident_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate multiple 2D Voigt kernels using scipy.special.wofz (Faddeeva function).

        Parameters
        ----------
        fwhm_rad : torch.Tensor
            FWHM values in radians, shape (N,)
        incident_angles : torch.Tensor
            Incident angles in degrees for each peak, shape (N,)

        Returns
        -------
        torch.Tensor
            Batch of normalized Voigt kernels with shape (N, 1, H, W)
        """
        # Convert FWHM from radians to pixels
        detector_distance = torch.linalg.norm(self.det_corner_0)
        incident_angles_rad = incident_angles * torch.pi / 180
        R = detector_distance / torch.cos(incident_angles_rad)

        # Convert to pixels
        gammas = (fwhm_rad * R / self.pixel_size_z) / 2  # HWHM
        sigma = self.gaussian_sigma

        # Compute radius based on maximum gamma
        max_gamma = gammas.max()
        threshold = self.kernel_threshold / 2

        radius = torch.tensor(1.0)
        while True:
            g_val = torch.exp(-(radius**2) / (2 * sigma**2))
            l_val = 1 / (1 + (radius / max_gamma) ** 2)
            if g_val < threshold and l_val < threshold:
                break
            radius = radius * 2

        # Create coordinate grid
        ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
        yy, zz = torch.meshgrid(ax, ax, indexing="ij")
        r = torch.sqrt(yy**2 + zz**2)

        # Pre-allocate output tensors
        kernels = []
        max_size = 0

        # Convert only the required arrays to numpy for scipy
        r_np = r.cpu().numpy()
        sigma_np = float(sigma)

        # Generate kernels for all peaks
        for gamma in gammas:
            # Use scipy.special.wofz for Voigt profile computation
            gamma_np = float(gamma)
            kernel = torch.from_numpy(voigt_profile(r_np, sigma_np, gamma_np)).to(
                gammas.device
            )

            kernel = kernel / kernel.sum()  # Normalize

            # Binary search for minimum size that retains required energy
            center = kernel.shape[0] // 2
            left, right = 1, center
            while left < right:
                mid = (left + right) // 2
                window = kernel[
                    center - mid : center + mid + 1, center - mid : center + mid + 1
                ]
                if window.sum() >= 1 - self.kernel_threshold:
                    right = mid
                else:
                    left = mid + 1

            # Crop and normalize
            r = left
            cropped = kernel[center - r : center + r + 1, center - r : center + r + 1]
            cropped = cropped / cropped.sum()

            kernels.append(cropped.unsqueeze(0).unsqueeze(0))
            max_size = max(max_size, 2 * r + 1)

        # Pad all kernels to max_size
        padded_kernels = []
        for k in kernels:
            if k.shape[-1] < max_size:
                pad = (max_size - k.shape[-1]) // 2
                k_padded = F.pad(k, (pad, pad, pad, pad), mode="constant", value=0)
                padded_kernels.append(k_padded)
            else:
                padded_kernels.append(k)

        # Combine kernels
        return torch.cat(padded_kernels, dim=0)

    def _deposit_kernels_batch(
        self,
        tensor: torch.Tensor,
        kernels: torch.Tensor,
        centers_z: torch.Tensor,  # z coordinate comes first
        centers_y: torch.Tensor,  # y coordinate comes second
    ) -> torch.Tensor:
        """Deposit multiple kernels onto a tensor using bilinear interpolation."""
        N = kernels.shape[0]
        kh, kw = kernels.shape[-2:]  # kernel height and width

        # Generate kernel grid offsets (z,y order)
        kz, ky = torch.meshgrid(
            torch.arange(kh, dtype=torch.float32) - (kh - 1) / 2,
            torch.arange(kw, dtype=torch.float32) - (kw - 1) / 2,
            indexing="ij",  # important: keep ij indexing for z,y order
        )

        # Make tensors contiguous and flatten
        kz = kz.flatten().repeat(N)  # z first
        ky = ky.flatten().repeat(N)  # y second

        # Ensure centers are contiguous
        centers_z = centers_z.repeat_interleave(kh * kw)
        centers_y = centers_y.repeat_interleave(kh * kw)

        # Get positions and weights (maintain z,y order)
        pos_z = centers_z + kz  # z coordinate
        pos_y = centers_y + ky  # y coordinate
        del centers_z, centers_y, kz, ky

        # Floor positions for interpolation
        z0 = torch.floor(pos_z).long()
        y0 = torch.floor(pos_y).long()
        dz = pos_z - z0.float()
        dy = pos_y - y0.float()
        del pos_z, pos_y

        # Calculate bilinear weights
        w00 = (1 - dz) * (1 - dy)
        w01 = (1 - dz) * dy
        w10 = dz * (1 - dy)
        w11 = dz * dy
        del dz, dy

        # Get kernel values and ensure contiguous memory
        kernel_values = kernels.reshape(N, -1).contiguous()
        k_flat = kernel_values.repeat_interleave(4, dim=0).flatten()
        del kernel_values

        # Stack positions and weights (maintain z,y order)
        zi = torch.stack([z0, z0, z0 + 1, z0 + 1]).flatten()
        yi = torch.stack([y0, y0 + 1, y0, y0 + 1]).flatten()
        weights = torch.stack([w00, w01, w10, w11]).flatten()
        del z0, y0, w00, w01, w10, w11

        # Filter valid positions
        valid = (zi >= 0) & (zi < tensor.shape[0]) & (yi >= 0) & (yi < tensor.shape[1])

        # Directly compute weighted values and deposit them
        tensor.index_put_(
            indices=(zi[valid], yi[valid]),  # Access valid indices directly
            values=(k_flat[valid] * weights[valid]),  # Compute product directly
            accumulate=True,
        )

        # Clean up intermediates
        del zi, yi, weights, k_flat, valid

        return tensor


def voigt_profile(r, sigma, gamma):
    """
    Compute Voigt profile using the Faddeeva function.

    Parameters
    ----------
    r : ndarray
        Radial distances
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter

    Returns
    -------
    ndarray
        Computed Voigt profile
    """
    z = (r + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
