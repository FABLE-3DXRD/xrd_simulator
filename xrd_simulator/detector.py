"""The detector module is used to represent a 2D area detector. After diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` has been computed, the detector can render
the scattering as a pixelated image via the :func:`xrd_simulator.detector.Detector.render`
function.

Here is a minimal example of how to instantiate a detector object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_detector.py

Below follows a detailed description of the detector class attributes and functions.

"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy.typing as npt
import dill
from scipy.special import wofz
import numpy as np
import torch
import torch.nn.functional as F

from xrd_simulator import utils
from xrd_simulator.utils import ensure_torch, ensure_numpy

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
        """Render diffraction frames using different peak profile methods.

        Available methods:
        - 'gauss': Fast Gaussian profiles, ideal for quick analysis and perfect crystals
        - 'voigt': Physically accurate profiles combining instrumental (Gaussian) and
                  crystal size (Lorentzian) effects. More computationally intensive.
        - 'volumes': Projects 3D crystal volumes, showing morphology and strain effects.
                    Requires volume data in peaks_dict.

        Args:
            peaks_dict: Peak data containing 'peaks' tensor and metadata
            frames_to_render: Number of frames to generate (0 = auto)
            method: Rendering method ('gauss', 'voigt', or 'volumes')

        Returns:
            torch.Tensor: Rendered diffraction frames (frames, height, width)

        Raises:
            ValueError: If invalid method specified
        """
        peaks_dict = self._peaks_detector_intersection(peaks_dict, frames_to_render)

        # Report crystallite size statistics
        peaks = peaks_dict["peaks"]
        crystallite_size = (
            2.0 * (3 * peaks[:, 21] / (4 * np.pi)) ** (1 / 3) * 1000  # microns to nanometers
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

        if method == "gauss":
            diffraction_frames = self._render_gauss_peaks(peaks_dict["peaks"])
        elif method == "voigt":
            diffraction_frames = self._render_voigt_peaks(peaks_dict["peaks"], crystallite_size)
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
        
        Peak tensor column mapping (after _peaks_detector_intersection):
            0-24: Original columns from polycrystal (grain_index through peak_index)
            25: zd (detector x-coordinate in pixels)
            26: yd (detector y-coordinate in pixels) 
            27: incident_angle (in degrees)
            28: frame (frame index for time-resolved rendering)
            29: intensity (combined with structure, polarization, lorentz factors)
        """
        pixel_indices = torch.cat(
            (
                ((peaks[:, 25]) / self.pixel_size_z).unsqueeze(1),  # Shifted by 1 due to large_grain column
                ((peaks[:, 26]) / self.pixel_size_y).unsqueeze(1),  # Shifted by 1
                peaks[:, 28].unsqueeze(1),  # Frame index
            ),
            dim=1,
        ).to(torch.int32)

        # Use column 28 (frame index) to determine number of frames, not column 27
        frames_n = int(peaks[:, 28].max().item()) + 1

        diffraction_frames = torch.zeros(
            (
                frames_n,
                self.pixel_coordinates.shape[0],
                self.pixel_coordinates.shape[1],
            ),
            device=peaks.device
        )

        # Get continuous pixel coordinates
        pos_z = peaks[:, 25] / self.pixel_size_z  # zd coordinate
        pos_y = peaks[:, 26] / self.pixel_size_y  # yd coordinate
        frame_idx = peaks[:, 28].long()  # Frame index

        # Get integer coordinates and fractions for interpolation
        z0 = pos_z.floor().long()
        y0 = pos_y.floor().long()
        dz = pos_z - z0.float()
        dy = pos_y - y0.float()

        # Calculate bilinear weights times intensities
        intensities = peaks[:, 29].unsqueeze(1)  # Shifted by 1 due to large_grain column
        weights = torch.stack([
            (1 - dz) * (1 - dy),  # w00
            (1 - dz) * dy,        # w01
            dz * (1 - dy),        # w10
            dz * dy               # w11
        ], dim=1) * intensities

        # Filter valid coordinates (within frame bounds)
        valid = (z0 >= 0) & (z0 < diffraction_frames.shape[1]-1) & \
               (y0 >= 0) & (y0 < diffraction_frames.shape[2]-1)
        
        if valid.any():
            z0 = z0[valid]
            y0 = y0[valid]
            frame_idx = frame_idx[valid]
            weights = weights[valid]

            # Use index_put_ for each corner with weights
            corners = [
                (z0, y0),           # bottom-left
                (z0, y0+1),         # bottom-right  
                (z0+1, y0),         # top-left
                (z0+1, y0+1),       # top-right
            ]
            
            for i, (zi, yi) in enumerate(corners):
                diffraction_frames.index_put_(
                    (frame_idx, zi, yi),
                    weights[:, i],
                    accumulate=True
                )

        # Apply Gaussian convolution
        diffraction_frames = self._conv2d_gaussian_kernel(diffraction_frames)

        return diffraction_frames

    def _render_voigt_peaks(self, peaks: torch.Tensor, crystallite_size: torch.Tensor) -> torch.Tensor:
        """Deposit Voigt kernels with intelligent batching to maximize VRAM utilization.

        Parameters
        ----------
        peaks : torch.Tensor
            Processed peaks tensor with precalculated intensities and frame indices
        crystallite_size : torch.Tensor
            Crystallite size in nanometers for batch size calculation

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames
        
        Peak tensor column mapping (after _peaks_detector_intersection):
            0-24: Original columns from polycrystal (grain_index through peak_index)
            25: zd (detector x-coordinate in pixels)
            26: yd (detector y-coordinate in pixels)
            27: incident_angle (in degrees)
            28: frame (frame index for time-resolved rendering)
            29: intensity (combined with structure, polarization, lorentz factors)
        """

        # Intelligent batch size calculation based on available VRAM
        if torch.cuda.is_available():
            # Get available GPU memory
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            free_memory = available_memory - cached_memory
            
            # Use up to 70% of free memory for batching (leaving margin for kernels and operations)
            target_memory_usage = free_memory * 0.7
            
            # Estimate memory per peak kernel (depends on kernel size)
            # A typical Voigt kernel might be ~100x100 pixels = 10K floats = ~80KB per peak
            avg_kernel_size = 100 * 100 * 8  # bytes (float64)
            batch_size = max(int((target_memory_usage * 1024**3) / avg_kernel_size), 100)
        else:
            # CPU fallback: use conservative batch size based on crystallite size
            batch_size = max(int(crystallite_size.min() * 100), 100)
        
        print(f"[Voigt Rendering] Batch size: {batch_size} peaks per batch")
        if torch.cuda.is_available():
            print(f"[Voigt Rendering] VRAM: {available_memory:.1f}GB total, {free_memory:.1f}GB free")
        
        frames_n = int(peaks[:, 28].max() + 1)
        frames = torch.zeros(
            (frames_n, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1])
        )

        for frame_idx in range(frames_n):
            frame_mask = peaks[:, 28] == frame_idx
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
                batch_count = end_idx - start_idx

                fwhm_rad = batch_peaks[:, 23]
                incident_angles = batch_peaks[:, 27]
                zd = batch_peaks[:, 25] / self.pixel_size_z
                yd = batch_peaks[:, 26] / self.pixel_size_y
                intensities = batch_peaks[:, 29]

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
                    f"Frame {frame_idx+1}: {progress}% complete ({batch_count} peaks in batch)"
                )

            print(f"Completed frame {frame_idx+1}/{frames_n}")

        return frames

    def _render_projected_volumes(
        self,
        peaks_dict: Dict[str, Union[torch.Tensor, List[str]]]
    ) -> torch.Tensor:
        """Render projected volumes onto detector frames using volume projection.
        This provides accurate peak shapes by projecting 3D crystal volumes.

        Args:
            peaks_dict: Dictionary containing peak information, convex hulls and metadata

        Returns:
            Rendered diffraction frames with projected volumes
        """
        convex_hulls = peaks_dict["convex_hulls"]
        frames_bundle = peaks_dict["peaks"][:, 28].unique()  # frame column is index 28 (24 original + large_grain + zd,yd,angle)
        device = peaks_dict["peaks"].device
        scattered_vectors = peaks_dict["scattered_vectors"]

        diffraction_frames = []
        for frame_index in frames_bundle:
            # Initialize frame on same device as peaks
            frames = torch.zeros(
                (self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=device
            )
            
            for i, hull in enumerate(convex_hulls):
                if hull is not None:
                    # Create minimal projection context
                    proj_context = {
                        'convex_hull': hull,
                        'scattered_wave_vector': scattered_vectors[i]
                    }
                    
                    # Project volume using only necessary data
                    box = self._get_projected_bounding_box(proj_context)
                    if box is not None:
                        # Keep projection in numpy for computation
                        projection = self.project_convex_hull(proj_context, box)
                        # Use pre-calculated intensity from peaks tensor
                        intensity = peaks_dict["peaks"][i, 29]  # intensity column is index 29 (24 original + large_grain + zd,yd,angle + frame)
                        
                        # Handle infinite values
                        if torch.isinf(intensity):
                            frames[box[0]:box[1], box[2]:box[3]] = float('inf')
                        else:
                            # Keep computation on device
                            projection_torch = ensure_torch(projection).to(device)
                            result = projection_torch * intensity
                            result = result * self.pixel_size_z * self.pixel_size_y
                            frames[box[0]:box[1], box[2]:box[3]] += result
            
            # Apply Gaussian convolution for point spread function
            frames = self._conv2d_gaussian_kernel(frames)
            diffraction_frames.append(frames)
            
        return torch.stack(diffraction_frames)

    def _conv2d_gaussian_kernel(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply the point spread function to the detector frames.
        
        Applies 2D convolution with a Gaussian kernel to each frame.
        Input tensor can be 2D (H,W) or 3D (N,H,W).
        Output has same dimensions as input after convolution.
        """
        frames = ensure_torch(frames)
        
        # Handle different input dimensions
        input_dims = frames.ndim
        if input_dims == 2:
            # Single frame (H,W) -> (1,1,H,W)
            frames = frames.unsqueeze(0).unsqueeze(0)
        elif input_dims == 3:
            # Multiple frames (N,H,W) -> (N,1,H,W)
            frames = frames.unsqueeze(1)
            
        # Prepare kernel for conv2d (1,1,kH,kW)
        kernel = self.gaussian_kernel.unsqueeze(0).unsqueeze(0)

        padding = kernel.shape[-1] // 2
        with torch.no_grad():
            output = torch.nn.functional.conv2d(frames, weight=kernel, padding=padding)
            
        # Restore original dimensions
        if input_dims == 2:
            output = output.squeeze(0).squeeze(0)  # (1,1,H,W) -> (H,W)
        elif input_dims == 3:
            output = output.squeeze(1)  # (N,1,H,W) -> (N,H,W)

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
        self, proj_context: Dict[str, Any], box: Tuple[int, int, int, int]
    ) -> npt.NDArray:
        """Project convex hull onto detector region.

        Args:
            proj_context: Dictionary containing convex_hull and scattered_wave_vector
            box: (min_z, max_z, min_y, max_y) bounds for projection

        Returns:
            Array of clip lengths between hull and detector rays
        """
        # Get pixel coordinates and ensure they're in numpy
        pixel_coords = self.pixel_coordinates[box[0]:box[1], box[2]:box[3], :]
        if pixel_coords.device.type == 'cuda':
            pixel_coords = pixel_coords.cpu()
        ray_points = ensure_numpy(pixel_coords).reshape(
            (box[1] - box[0]) * (box[3] - box[2]), 3
        )

        # Keep plane calculations in numpy
        hull = proj_context['convex_hull']
        plane_normals = ensure_numpy(hull.equations[:, 0:3])
        plane_ofsets = ensure_numpy(hull.equations[:, 3]).reshape(
            hull.equations.shape[0], 1
        )
        plane_points = -np.multiply(plane_ofsets, plane_normals)

        # Keep ray direction calculation in numpy
        scattered_vec = ensure_numpy(proj_context['scattered_wave_vector'])
        ray_direction = scattered_vec / np.linalg.norm(scattered_vec)

        # Ensure contiguous arrays for calculations
        ray_points = np.ascontiguousarray(ray_points)
        ray_direction = np.ascontiguousarray(ray_direction)
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
        # Convert factors to numpy for consistency with convex hull calculations
        intensity_factor = 1.0
        if lorentz:
            intensity_factor *= ensure_numpy(scattering_unit.lorentz_factor)
        if polarization:
            intensity_factor *= ensure_numpy(scattering_unit.polarization_factor)
        if structure_factor:
            if scattering_unit.phase.structure_factors is not None:
                real_sf = ensure_numpy(scattering_unit.real_structure_factor)
                imag_sf = ensure_numpy(scattering_unit.imaginary_structure_factor)
                intensity_factor *= real_sf**2 + imag_sf**2
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
        self, proj_context: Dict[str, Any]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute bounding detector pixel indices of the bounding the projection of a convex hull.

        Args:
            proj_context: Dictionary containing convex_hull and scattered_wave_vector

        Returns:
            (:obj=`tuple` of :obj=`int`) indices that can be used to slice the detector frame array and get the pixels that
                are within the bounding box.

        """
        # Keep vertices in numpy since they come from convex hull
        hull = proj_context['convex_hull']
        vertices = hull.points[hull.vertices]

        # Convert to torch for intersection calculation
        # Reshape scattered_wave_vector to (N,3) by repeating for each vertex
        scattered_vec = ensure_torch(proj_context['scattered_wave_vector'])
        scattered_vec = scattered_vec.unsqueeze(0).repeat(vertices.shape[0], 1)
        vertices_torch = ensure_torch(vertices)
        
        # Get intersections and ensure they're on CPU before numpy conversion
        projected_vertices = self.get_intersection(scattered_vec, vertices_torch)
        if projected_vertices.device.type == 'cuda':
            projected_vertices = projected_vertices.cpu()
        projected_vertices = ensure_numpy(projected_vertices)

        # Convert zmax/ymax to CPU scalars for bounds checking
        zmax_np = float(ensure_numpy(self.zmax))
        ymax_np = float(ensure_numpy(self.ymax))

        # Calculate bounds using numpy operations
        min_zd = max(float(np.nanmin(projected_vertices[:, 0])), 0)
        max_zd = min(float(np.nanmax(projected_vertices[:, 0])), zmax_np)
        min_yd = max(float(np.nanmin(projected_vertices[:, 1])), 0)
        max_yd = min(float(np.nanmax(projected_vertices[:, 1])), ymax_np)

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

        # Filter peaks by detector bounds
        mask = self.contains(peaks[:, 25], peaks[:, 26])  # Shifted by 1 due to large_grain column
        peaks = peaks[mask]
        
        # Filter auxiliary data by same mask if present
        if "convex_hulls" in peaks_dict:
            peaks_dict["convex_hulls"] = [h for i, h in enumerate(peaks_dict["convex_hulls"]) if mask[i]]
        if "scattered_vectors" in peaks_dict:
            peaks_dict["scattered_vectors"] = peaks_dict["scattered_vectors"][mask]

        time = peaks[:, 6].contiguous()
        
        # Handle frame assignment: when frames_to_render==1, assign all to frame 0
        if frames_to_render <= 1:
            frame = torch.zeros((peaks.shape[0], 1), dtype=torch.int64, device=peaks.device)
        else:
            bins = torch.linspace(0, 1, frames_to_render).contiguous()
            frame = torch.bucketize(time, bins).unsqueeze(1) - 1
            # Clamp frame indices to valid range [0, frames_to_render-1]
            frame = torch.clamp(frame, 0, frames_to_render - 1)
        
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
                # Convert numpy array to torch tensor and move to correct device immediately
                kernel = ensure_torch(voigt_profile(r_np, sigma_np, gamma_np))
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
                padded_kernels.append(k_padded.to(device=gammas.device))
            else:
                padded_kernels.append(k.to(device=gammas.device))

        # Ensure all kernels are on the same device before concatenating
        result = torch.cat(padded_kernels, dim=0)
        return result

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

    def get_vram_info(self) -> Dict[str, float]:
        """Get current VRAM usage information.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with keys:
                - 'total_gb': Total GPU memory in GB
                - 'allocated_gb': Currently allocated memory in GB
                - 'reserved_gb': Currently reserved memory in GB
                - 'free_gb': Free memory available in GB
                - 'utilization_percent': Percentage of total memory in use
        """
        if not torch.cuda.is_available():
            return {
                'total_gb': 0,
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0,
                'utilization_percent': 0
            }
        
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - reserved
        utilization = (allocated / total_memory) * 100
        
        return {
            'total_gb': total_memory,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free_memory,
            'utilization_percent': utilization
        }
    
    def print_vram_info(self) -> None:
        """Print formatted VRAM usage information."""
        info = self.get_vram_info()
        if info['total_gb'] == 0:
            print("CUDA not available - running on CPU")
            return
        
        print("\n" + "="*60)
        print("GPU MEMORY STATUS")
        print("="*60)
        print(f"Total VRAM:      {info['total_gb']:>8.1f} GB")
        print(f"Allocated:       {info['allocated_gb']:>8.1f} GB")
        print(f"Reserved:        {info['reserved_gb']:>8.1f} GB")
        print(f"Free:            {info['free_gb']:>8.1f} GB")
        print(f"Utilization:     {info['utilization_percent']:>7.1f}%")
        print("="*60 + "\n")

