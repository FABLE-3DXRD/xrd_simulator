"""The detector module is used to represent a 2D area detector. After diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` has been computed, the detector can render
the scattering as a pixelated image via the :func:`xrd_simulator.detector.Detector.render`
function.

Here is a minimal example of how to instantiate a detector object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_detector.py

Below follows a detailed description of the detector class attributes and functions.

"""

from typing import Dict, List, Optional, Tuple, Union
import numpy.typing as npt
import dill
from scipy.special import wofz
import numpy as np
import torch
import torch.nn.functional as F

from xrd_simulator import utils
from xrd_simulator.utils import ensure_torch
from xrd_simulator.cuda import device

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
        gaussian_sigma (:obj=`float`, optional): Standard deviation of the Gaussian point spread function in pixels.
            Defaults to 1.0.
        kernel_threshold (:obj=`float`, optional): Intensity threshold used to determine the kernel size.
            The kernel will extend until the Gaussian tail drops below this value. Defaults to 0.05.

    Attributes:
        pixel_size_z (:obj=`float`): Pixel side length along zdhat (rectangular pixels) in units of microns.
        pixel_size_y (:obj=`float`): Pixel side length along ydhat (rectangular pixels) in units of microns.
        det_corner_0,det_corner_1,det_corner_2 (:obj=`numpy array`): Detector corner 3d coordinates ``shape=(3,)``.
            The origin of the detector is at det_corner_0.
        frames (:obj=`list` of :obj=`list` of :obj=`scattering_unit.ScatteringUnit`): Analytical diffraction frames
        zdhat,ydhat (:obj=`numpy array`): Detector basis vectors.
        normal (:obj=`numpy array`): Detector normal, formed as the cross product: numpy.cross(self.zdhat, self.ydhat)
        zmax,ymax (:obj=`numpy array`): Detector width and height.
        pixel_coordinates  (:obj=`numpy array`): Real space 3d detector pixel coordinates. ``shape=(n,3)``
        gaussian_sigma (:obj=`float`): Standard deviation of the Gaussian point spread function.
        kernel_threshold (:obj=`float`): Threshold value that determines the kernel size.
        gaussian_kernel (:obj=`torch.Tensor`): Pre-computed normalized 2D Gaussian kernel.

    """

    def __init__(
        self,
        pixel_size_z: float,
        pixel_size_y: float,
        det_corner_0: npt.NDArray,
        det_corner_1: npt.NDArray,
        det_corner_2: npt.NDArray,
        gaussian_sigma: float = 1.0,
        kernel_threshold: float = 0.05,
        use_lorentz: bool = True,
        use_polarization: bool = True,
        use_structure_factor: bool = True,
    ):
        self.det_corner_0 = ensure_torch(det_corner_0).to(device)
        self.det_corner_1 = ensure_torch(det_corner_1).to(device)
        self.det_corner_2 = ensure_torch(det_corner_2).to(device)

        self.pixel_size_z = ensure_torch(pixel_size_z).to(device)
        self.pixel_size_y = ensure_torch(pixel_size_y).to(device)

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
        self.gaussian_kernel = self._generate_gaussian_kernel()

        self.lorentz_factor = use_lorentz
        self.polarization_factor = use_polarization
        self.structure_factor = use_structure_factor

    def render(
        self,
        peaks_dict: Dict[str, Union[torch.Tensor, List[str]]],
        frames_to_render: int = 0,
        method: str = "centroid",
    ) -> torch.Tensor:
        """Render diffraction frames from peak data.

        Args:
            peaks_dict: Dictionary containing peak information and metadata
            frames_to_render: Number of frames to generate (0 = auto)
            method: Rendering method, one of: 'gauss', 'voigt', or 'volumes'

        Returns:
            Rendered diffraction frames tensor

        Raises:
            ValueError: If invalid rendering method specified
        """
        peaks_dict = self._peaks_detector_intersection(peaks_dict, frames_to_render)

        if method == "centroid":
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

    def _render_gauss_peaks(self, peaks: torch.Tensor) -> torch.Tensor:
        """Render Gaussian peaks onto detector frames.

        Args:
            peaks: Peak data tensor containing positions and intensities

        Returns:
            Rendered diffraction frames with Gaussian peaks
        """
        pixel_indices = torch.cat(
            (
                ((peaks[:, 24]) / self.pixel_size_z).unsqueeze(1),
                ((peaks[:, 25]) / self.pixel_size_y).unsqueeze(1),
                peaks[:, 27].unsqueeze(1),
            ),
            dim=1,
        ).to(torch.int32)

        frames_n = peaks[:, 27].unique().shape[0]

        diffraction_frames = torch.zeros(
            (
                frames_n,
                self.pixel_coordinates.shape[0],
                self.pixel_coordinates.shape[1],
            )
        )

        unique_coords, inverse_indices = torch.unique(
            pixel_indices, dim=0, return_inverse=True
        )

        counts = torch.bincount(inverse_indices, weights=peaks[:, 28])

        result = torch.cat((unique_coords, counts.unsqueeze(1)), dim=1).type_as(
            diffraction_frames
        )

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

                end_idx = min(start_idx + batch_size, len(frame_peaks))
                batch_peaks = frame_peaks[start_idx:end_idx]

                fwhm_rad = batch_peaks[:, 23]
                incident_angles = batch_peaks[:, 26]
                zd = batch_peaks[:, 24] / self.pixel_size_z
                yd = batch_peaks[:, 25] / self.pixel_size_y
                intensities = batch_peaks[:, 28]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                kernels = self._voigt_kernel_batch(fwhm_rad, incident_angles)
                kernels = kernels * intensities.view(-1, 1, 1, 1)
                frames[frame_idx] = self._deposit_kernels_batch(
                    frames[frame_idx], kernels, zd, yd
                )

                del kernels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                start_idx = end_idx

                progress = int((start_idx / len(frame_peaks)) * 100)
                print(
                    f"Frame {frame_idx+1}: {progress}% complete (batch size: {batch_size})"
                )

            print(f"Completed frame {frame_idx+1}/{frames_n}")

        return frames

    def _render_projected_volumes(
        self,
        peaks_dict: Dict[str, Union[torch.Tensor, List[str]]],
        renderer: callable,
    ) -> npt.NDArray:
        """Render projected volumes onto detector frames.

        Args:
            peaks_dict: Dictionary containing peak information and metadata
            renderer: Function to render scattering units

        Returns:
            Rendered diffraction frames with projected volumes
        """
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

    def _conv2d_gaussian_kernel(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply the point spread function to the detector frames."""
        frames = ensure_torch(frames)
        if frames.ndim == 2:
            frames = frames.unsqueeze(0)  # Add channel dimension if only 1 image
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)  # Add channel dimension if only 1 image

        # Add required dimensions for conv2d (N,C,H,W)
        kernel = self.gaussian_kernel.unsqueeze(0).unsqueeze(0)

        padding = kernel.shape[-1] // 2
        with torch.no_grad():
            output = torch.nn.functional.conv2d(frames, weight=kernel, padding=padding)

        return output

    def get_intersection(
        self, ray_direction: torch.Tensor, source_point: torch.Tensor
    ) -> torch.Tensor:
        """Get detector intersection coordinates for rays.

        Args:
            ray_direction: Direction vectors for each ray, shape (N,3)
            source_point: Origin points for each ray, shape (N,3)

        Returns:
            Intersection coordinates (zd,yd) and incident angles, shape (N,3)
        """
        s = torch.matmul(self.det_corner_0 - source_point, self.normal) / torch.matmul(
            ray_direction, self.normal
        )

        intersection = source_point + ray_direction * s.unsqueeze(1)
        intersection[s < 0] = np.nan
        zd = torch.matmul(intersection - self.det_corner_0, self.zdhat)
        yd = torch.matmul(intersection - self.det_corner_0, self.ydhat)

        ray_dir_norm = ray_direction / torch.norm(ray_direction, dim=1).unsqueeze(-1)
        normal_norm = self.normal / torch.linalg.norm(self.normal)
        cosine_theta = torch.matmul(ray_dir_norm, -normal_norm)
        incident_angle_deg = torch.arccos(cosine_theta) * (180 / torch.pi)
        return torch.stack((zd, yd, incident_angle_deg), dim=1)

    def contains(self, zd: torch.Tensor, yd: torch.Tensor) -> torch.Tensor:
        """Check if detector coordinates are within bounds.

        Args:
            zd: Z coordinates to check
            yd: Y coordinates to check

        Returns:
            Boolean mask of valid coordinates
        """
        return (zd >= 0) & (zd <= self.zmax) & (yd >= 0) & (yd <= self.ymax)

    def project_convex_hull(
        self, scattering_unit: "ScatteringUnit", box: Tuple[int, int, int, int]
    ) -> npt.NDArray:
        """Project scattering unit's convex hull onto detector region.

        Args:
            scattering_unit: Unit to project
            box: (min_z, max_z, min_y, max_y) bounds for projection

        Returns:
            Array of clip lengths between hull and detector rays
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

    def get_wrapping_cone(
        self, k: torch.Tensor, source_point: torch.Tensor
    ) -> torch.Tensor:
        """Compute cone that wraps detector corners around wavevector.

        Args:
            k: Central wavevector of cone, shape (3,)
            source_point: Cone vertex point, shape (3,)

        Returns:
            Half-angle of cone opening in radians
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
        )
        return torch.max(cone_opening) / 2.0

    def save(self, path: str) -> None:
        """Save detector to disk.

        Args:
            path: Output file path (.det extension added if missing)
        """
        if not path.endswith(".det"):
            path = path + ".det"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Detector":
        """Load detector from disk.

        Args:
            path: Path to .det file

        Returns:
            Loaded Detector instance

        Raises:
            ValueError: If file extension is not .det
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

    def _generate_gaussian_kernel(self) -> torch.Tensor:
        """
        Generates a normalized 2D Gaussian kernel with dynamic size based on intensity threshold.

        Returns:
            torch.Tensor: 2D Gaussian kernel of shape (H, W).
        """
        sigma = ensure_torch(self.gaussian_sigma)
        threshold = self.kernel_threshold

        radius = 1
        while True:
            value = torch.exp(-(radius**2) / (2 * sigma**2))
            if value < threshold / 2:
                break
            radius += 1

        kernel_size = 2 * radius + 1

        ax = torch.arange(kernel_size, dtype=torch.float64) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel  # Return as (H,W) instead of (1,1,H,W)

    def _get_pixel_coordinates(self) -> torch.Tensor:
        """Calculate real-space coordinates for each detector pixel.

        Returns:
            Tensor of shape (Z,Y,3) containing pixel coordinates
        """
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
        self,
        scattering_unit: "ScatteringUnit",
        frame: npt.NDArray,
        lorentz: bool,
        polarization: bool,
        structure_factor: bool,
    ) -> None:
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
        self,
        scattering_unit: "ScatteringUnit",
        lorentz: bool,
        polarization: bool,
        structure_factor: bool,
    ) -> float:
        """Calculate combined intensity scaling factor.

        Args:
            scattering_unit: Unit to get factors for
            lorentz: Whether to include Lorentz factor
            polarization: Whether to include polarization
            structure_factor: Whether to include structure factor

        Returns:
            Combined intensity scaling factor

        Raises:
            ValueError: If structure factors needed but not available
        """
        intensity_factor = 1.0
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

    def _detector_coordinate_to_pixel_index(
        self, zd: float, yd: float
    ) -> Tuple[int, int]:
        row_index = int(zd / self.pixel_size_z)
        col_index = int(yd / self.pixel_size_y)
        return row_index, col_index

    def _get_projected_bounding_box(
        self, scattering_unit: "ScatteringUnit"
    ) -> Optional[Tuple[int, int, int, int]]:
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

    def _peaks_detector_intersection(
        self,
        peaks_dict: Dict[str, Union[torch.Tensor, List[str]]],
        frames_to_render: int,
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
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

        zd_yd_angle = self.get_intersection(peaks[:, 13:16], peaks[:, 16:19])
        peaks = torch.cat((peaks, zd_yd_angle), dim=1)
        peaks_dict["columns"].extend(["zd", "yd", "incident_angle"])

        mask = self.contains(peaks[:, 24], peaks[:, 25])
        peaks = peaks[mask]

        time = peaks[:, 6].contiguous()
        bins = torch.linspace(0, 1, frames_to_render).contiguous()
        frame = torch.bucketize(time, bins).unsqueeze(1) - 1
        peaks = torch.cat((peaks, frame), dim=1)
        peaks_dict["columns"].append("frame")

        intensity = peaks[:, 21]
        if self.structure_factor:
            intensity = intensity * peaks[:, 5]
        if self.polarization_factor:
            intensity = intensity * peaks[:, 20]
        if self.lorentz_factor:
            intensity = intensity * peaks[:, 19]

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
        detector_distance = torch.linalg.norm(self.det_corner_0)
        incident_angles_rad = incident_angles * torch.pi / 180
        R = detector_distance / torch.cos(incident_angles_rad)

        gammas = (fwhm_rad * R / self.pixel_size_z) / 2
        sigma = self.gaussian_sigma

        max_gamma = gammas.max()
        threshold = self.kernel_threshold / 2

        radius = torch.tensor(1.0)
        while True:
            g_val = torch.exp(-(radius**2) / (2 * sigma**2))
            l_val = 1 / (1 + (radius / max_gamma) ** 2)
            if g_val < threshold and l_val < threshold:
                break
            radius = radius * 2

        ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
        yy, zz = torch.meshgrid(ax, ax, indexing="ij")
        r = torch.sqrt(yy**2 + zz**2)

        kernels = []
        max_size = 0

        def voigt_profile(r: npt.NDArray, sigma: float, gamma: float) -> npt.NDArray:
            """Compute Voigt profile using Faddeeva function.

            Args:
                r: Radial distances
                sigma: Gaussian width parameter
                gamma: Lorentzian width parameter

            Returns:
                Computed Voigt profile values
            """
            z = (r + 1j * gamma) / (sigma * np.sqrt(2))
            return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

        r_np = r.cpu().numpy()
        sigma_np = float(sigma)

        for gamma in gammas:
            gamma_np = float(gamma)
            if gamma_np / self.gaussian_sigma < 0.01:
                kernel = self.gaussian_kernel
            else:
                kernel = torch.from_numpy(voigt_profile(r_np, sigma_np, gamma_np))

            kernel = kernel / kernel.sum()

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

            r = left
            cropped = kernel[center - r : center + r + 1, center - r : center + r + 1]
            cropped = cropped / cropped.sum()

            kernels.append(cropped.unsqueeze(0).unsqueeze(0))
            max_size = max(max_size, 2 * r + 1)

        padded_kernels = []
        for k in kernels:
            if k.shape[-1] < max_size:
                pad = (max_size - k.shape[-1]) // 2
                k_padded = F.pad(k, (pad, pad, pad, pad), mode="constant", value=0)
                padded_kernels.append(k_padded)
            else:
                padded_kernels.append(k)

        return torch.cat(padded_kernels, dim=0)

    def _deposit_kernels_batch(
        self,
        tensor: torch.Tensor,
        kernels: torch.Tensor,
        centers_z: torch.Tensor,
        centers_y: torch.Tensor,
    ) -> torch.Tensor:
        """Deposit multiple kernels onto a tensor using bilinear interpolation.

        Args:
            tensor: Target tensor to deposit onto
            kernels: Batch of kernels to deposit, shape (N,1,H,W)
            centers_z: Z coordinates for kernel centers
            centers_y: Y coordinates for kernel centers

        Returns:
            Updated tensor with deposited kernels
        """
        N = kernels.shape[0]
        kh, kw = kernels.shape[-2:]

        kz, ky = torch.meshgrid(
            torch.arange(kh, dtype=torch.float32) - (kh - 1) / 2,
            torch.arange(kw, dtype=torch.float32) - (kw - 1) / 2,
            indexing="ij",
        )

        kz = kz.flatten().repeat(N)
        ky = ky.flatten().repeat(N)

        centers_z = centers_z.repeat_interleave(kh * kw)
        centers_y = centers_y.repeat_interleave(kh * kw)

        pos_z = centers_z + kz
        pos_y = centers_y + ky
        del centers_z, centers_y, kz, ky

        z0 = torch.floor(pos_z).long()
        y0 = torch.floor(pos_y).long()
        dz = pos_z - z0.float()
        dy = pos_y - y0.float()
        del pos_z, pos_y

        w00 = (1 - dz) * (1 - dy)
        w01 = (1 - dz) * dy
        w10 = dz * (1 - dy)
        w11 = dz * dy
        del dz, dy

        kernel_values = kernels.reshape(N, -1).contiguous()
        k_flat = kernel_values.repeat_interleave(4, dim=0).flatten()
        del kernel_values

        zi = torch.stack([z0, z0, z0 + 1, z0 + 1]).flatten()
        yi = torch.stack([y0, y0 + 1, y0, y0 + 1]).flatten()
        weights = torch.stack([w00, w01, w10, w11]).flatten()
        del z0, y0, w00, w01, w10, w11

        valid = (zi >= 0) & (zi < tensor.shape[0]) & (yi >= 0) & (yi < tensor.shape[1])

        tensor.index_put_(
            indices=(zi[valid], yi[valid]),
            values=(k_flat[valid] * weights[valid]),
            accumulate=True,
        )

        del zi, yi, weights, k_flat, valid

        return tensor
