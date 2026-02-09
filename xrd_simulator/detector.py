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
from scipy.special import j1
import numpy as np
import torch
import torch.nn.functional as F

from xrd_simulator import utils
from xrd_simulator.cuda import get_selected_device
from xrd_simulator.utils import ensure_torch, ensure_numpy, return_device_memory

torch.set_default_dtype(torch.float64)


class Detector:
    """Represents a rectangular 2D area detector.

    The detector collects :class:`xrd_simulator.scattering_unit.ScatteringUnit`
    during diffraction from a :class:`xrd_simulator.polycrystal.Polycrystal`.
    The detector implements various rendering of the scattering as 2D pixelated
    frames. The detector geometry is described by specifying the locations of
    three detector corners. The detector is described in the laboratory
    coordinate system.

    Parameters
    ----------
    det_corner_0 : numpy.ndarray
        Detector corner 3D coordinates, shape ``(3,)``. The origin of the
        detector is at this corner.
    det_corner_1 : numpy.ndarray
        Detector corner 3D coordinates, shape ``(3,)``. Defines the y-axis
        direction of the detector.
    det_corner_2 : numpy.ndarray
        Detector corner 3D coordinates, shape ``(3,)``. Defines the z-axis
        direction of the detector.
    n_pixels : int or tuple, optional
        Number of pixels. Can be a single int (square detector) or tuple
        ``(n_z, n_y)``. If provided, pixel sizes are calculated automatically
        from detector dimensions. Mutually exclusive with ``pixel_size``.
    pixel_size : float or tuple, optional
        Pixel side length in microns. Can be a single value (square pixels)
        or tuple ``(pixel_size_z, pixel_size_y)`` for rectangular pixels.
        ``zdhat`` is the unit vector from ``det_corner_0`` towards
        ``det_corner_2``. ``ydhat`` is the unit vector from ``det_corner_0``
        towards ``det_corner_1``.
    gaussian_sigma : float, optional
        Standard deviation of the Gaussian point spread function in pixels.
        Default is 1.0.
    kernel_threshold : float, optional
        Intensity threshold used to determine the kernel size. The kernel will
        extend until the Gaussian tail drops below this value. Default is 0.02.
    use_lorentz : bool, optional
        Whether to apply the Lorentz factor correction. Default is ``True``.
    use_polarization : bool, optional
        Whether to apply the polarization factor correction. Default is ``True``.
    use_structure_factor : bool, optional
        Whether to apply the structure factor. Default is ``True``.

    Attributes
    ----------
    pixel_size_z : float
        Pixel side length along ``zdhat`` (rectangular pixels) in microns.
    pixel_size_y : float
        Pixel side length along ``ydhat`` (rectangular pixels) in microns.
    det_corner_0 : torch.Tensor
        Detector corner 3D coordinates, shape ``(3,)``. The origin of the
        detector.
    det_corner_1 : torch.Tensor
        Detector corner 3D coordinates, shape ``(3,)``.
    det_corner_2 : torch.Tensor
        Detector corner 3D coordinates, shape ``(3,)``.
    frames : list
        Analytical diffraction frames.
    zdhat : torch.Tensor
        Detector basis vector along the z-axis direction.
    ydhat : torch.Tensor
        Detector basis vector along the y-axis direction.
    normal : torch.Tensor
        Detector normal, formed as the cross product:
        ``torch.cross(self.zdhat, self.ydhat)``.
    zmax : torch.Tensor
        Detector width along ``zdhat``.
    ymax : torch.Tensor
        Detector height along ``ydhat``.
    pixel_coordinates : torch.Tensor
        Real space 3D detector pixel coordinates, shape ``(n_z, n_y, 3)``.
    gaussian_sigma : float
        Standard deviation of the Gaussian point spread function.
    kernel_threshold : float
        Threshold value that determines the kernel size.
    gaussian_kernel : torch.Tensor
        Pre-computed normalized 2D Gaussian kernel.
    lorentz_factor : bool
        Whether the Lorentz factor correction is applied.
    polarization_factor : bool
        Whether the polarization factor correction is applied.
    structure_factor : bool
        Whether the structure factor is applied.
    """
    # ------------------------------------------------------------------
    # 1. Core lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        det_corner_0: npt.NDArray,
        det_corner_1: npt.NDArray,
        det_corner_2: npt.NDArray,
        n_pixels: int | tuple = None,
        pixel_size: float | tuple = None,
        gaussian_sigma: float = 1.0,
        kernel_threshold: float = 0.02,
        use_lorentz: bool = True,
        use_polarization: bool = True,
        use_structure_factor: bool = True,
    ):
        self.det_corner_0 = ensure_torch(det_corner_0)
        self.det_corner_1 = ensure_torch(det_corner_1)
        self.det_corner_2 = ensure_torch(det_corner_2)

        self.zmax = torch.linalg.norm(self.det_corner_2 - self.det_corner_0)
        self.ymax = torch.linalg.norm(self.det_corner_1 - self.det_corner_0)

        # Determine pixel sizes from either n_pixels or pixel_size
        if n_pixels is not None:
            if pixel_size is not None:
                raise ValueError("Cannot specify both n_pixels and pixel_size")
            if isinstance(n_pixels, (tuple, list)):
                n_z, n_y = n_pixels
            else:
                n_z = n_y = n_pixels
            self.pixel_size_z = self.zmax / n_z
            self.pixel_size_y = self.ymax / n_y
        elif pixel_size is not None:
            if isinstance(pixel_size, (tuple, list)):
                self.pixel_size_z = ensure_torch(pixel_size[0])
                self.pixel_size_y = ensure_torch(pixel_size[1])
            else:
                self.pixel_size_z = ensure_torch(pixel_size)
                self.pixel_size_y = ensure_torch(pixel_size)
        else:
            raise ValueError("Must specify either n_pixels=(n_z, n_y) or pixel_size")

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

    def save(self, path: str) -> None:
        """Save detector to disk.

        Parameters
        ----------
        path : str
            Output file path. The ``.det`` extension is added if missing.
        """
        if not path.endswith(".det"):
            path = path + ".det"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Detector":
        """Load detector from disk.

        Parameters
        ----------
        path : str
            Path to ``.det`` file.

        Returns
        -------
        Detector
            Loaded Detector instance.

        Raises
        ------
        ValueError
            If file extension is not ``.det``.
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

    # ------------------------------------------------------------------
    # 2. High-level public API
    # ------------------------------------------------------------------

    def render(
        self,
        peaks_dict: Dict[str, Union[torch.Tensor, List[str]]],
        frames_to_render: int = 1,
        method: str = "auto",
        macro_grain_limit = None, #depends on the detector pixel size
        micro_grain_limit = 0.1**3, #in cubic microns
        render_dtype: torch.dtype | None = None,
        verbose: bool | None = None,
    ) -> torch.Tensor:
        """Render diffraction frames using different peak profile methods.

        Available methods (named by grain size regime):

        - ``'micro'``: Fast Gaussian profiles, ideal for quick analysis and
          medium-sized grains.
        - ``'nano'``: Physically accurate Airy disk patterns from crystallite
          diffraction. Broader for smaller crystallites, delta-like for large
          crystals. This is the most realistic shape for size-broadened peaks.
        - ``'macro'``: Projects 3D crystal volumes, showing morphology and strain
          effects. Requires volume data in ``peaks_dict``.
        - ``'auto'``: Automatically selects the rendering method based on grain
          size, using ``macro_grain_limit`` and ``micro_grain_limit`` thresholds.

        Parameters
        ----------
        peaks_dict : dict
            Peak data dictionary containing ``'peaks'`` tensor and metadata
            columns.
        frames_to_render : int, optional
            Number of frames to generate. The diffraction time domain ``[0, 1]``
            is divided into ``frames_to_render`` equal bins, and each peak is
            assigned to the frame corresponding to its diffraction time. When
            set to 1, all peaks are rendered into a single integrated frame.
            Values less than 1 are clamped to 1. Default is 1.
        method : str, optional
            Rendering method: ``'micro'``, ``'nano'``, ``'macro'``, or ``'auto'``.
            Default is ``'micro'``.
        macro_grain_limit : float, optional
            Volume threshold (cubic microns) above which grains use the macro
            rendering method. Default is ``pixel_size**3``.
        micro_grain_limit : float, optional
            Volume threshold (cubic microns) below which grains use the nano
            rendering method. Default is ``0.001`` (0.1³ cubic microns).
        render_dtype : torch.dtype, optional
            Data type for rendered frames. If ``None``, uses the dtype of the
            peaks tensor.
        verbose : bool or None, optional
            If ``True``, print progress messages during rendering. If ``None``,
            inherits the ``verbose`` flag from ``peaks_dict`` (set by
            ``polycrystal.diffract``). Defaults to ``True`` if neither is set.

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames with shape ``(frames, height, width)``.

        Raises
        ------
        ValueError
            If an invalid method is specified.
        """

        peaks = peaks_dict["peaks"]
        columns = peaks_dict["columns"]
        beam = peaks_dict.get("beam", None)
        lab_mesh = peaks_dict.get("mesh_lab", None)
        rigid_body_motion = peaks_dict.get("rigid_body_motion", None)
        target_dtype = render_dtype if render_dtype is not None else peaks.dtype
        if verbose is None:
            verbose = peaks_dict.get("verbose", True)
        self._verbose = verbose  # Store for use in rendering methods
        
        if frames_to_render < 1:
            frames_to_render = 1
            if self._verbose:
                print(f"[Detector.render] frames_to_render < 1, setting to 1")

        peaks, columns = self._peaks_detector_intersection(peaks,columns, frames_to_render)

        if method == "macro":
            diffraction_frames = self._render_projected_volumes(
                peaks, beam, lab_mesh, rigid_body_motion, frames_to_render, target_dtype
            )
        elif method == "micro":
            diffraction_frames = self._render_gauss_peaks(peaks, frames_to_render, target_dtype)
        elif method == "nano":
            diffraction_frames = self._render_airy_peaks(peaks, frames_to_render, target_dtype)
        elif method == "auto":
            if macro_grain_limit is None:
                macro_grain_limit = (min(self.pixel_size_y, self.pixel_size_z))**3
            large_grains_mask = peaks[:,21] >= macro_grain_limit
            small_grains_mask = peaks[:,21] < micro_grain_limit
            medium_grains_mask = (~large_grains_mask) & (~small_grains_mask)

            diffraction_frames = torch.zeros(
                (frames_to_render, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=peaks.device,
            )

            # Collect contributions (each returns frames_to_render-length tensors)
            contrib_macro = self._render_projected_volumes(
                peaks[large_grains_mask], beam, lab_mesh, rigid_body_motion, frames_to_render, target_dtype
            )
            contrib_micro = self._render_gauss_peaks(
                peaks[medium_grains_mask], frames_to_render, target_dtype
            )
            contrib_nano = self._render_airy_peaks(
                peaks[small_grains_mask], frames_to_render, target_dtype
            )

            # Ensure dtype/device matching and add up
            diffraction_frames = diffraction_frames.to(contrib_nano.device)
            diffraction_frames += contrib_nano
            diffraction_frames += contrib_macro
            diffraction_frames += contrib_micro
        else:
            raise ValueError(
                f"Invalid method: {method}. Must be one of: 'micro', 'nano', 'macro', or 'auto'"
            )

        return diffraction_frames

    def contains(self, zd: torch.Tensor, yd: torch.Tensor) -> torch.Tensor:
        """Check if detector coordinates are within bounds.

        Parameters
        ----------
        zd : torch.Tensor
            Z-axis detector coordinates.
        yd : torch.Tensor
            Y-axis detector coordinates.

        Returns
        -------
        torch.Tensor
            Boolean tensor indicating which coordinates are within detector
            bounds.
        """
        return (zd >= 0) & (zd <= self.zmax) & (yd >= 0) & (yd <= self.ymax)

    # ------------------------------------------------------------------
    # 3. Geometry & coordinate helpers
    # ------------------------------------------------------------------

    def _get_pixel_coordinates(self) -> torch.Tensor:
        """Calculate real-space coordinates for each detector pixel.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(Z, Y, 3)`` containing 3D coordinates for each
            pixel center.
        """
        # Get device from det_corner_0 to ensure all tensors are on same device
        dev = self.det_corner_0.device
        
        zds = torch.arange(0, self.zmax, self.pixel_size_z, device=dev)
        yds = torch.arange(0, self.ymax, self.pixel_size_y, device=dev)
        Z, Y = torch.meshgrid(zds, yds, indexing="ij")
        Zds = torch.zeros((len(zds), len(yds), 3), device=dev)
        Yds = torch.zeros((len(zds), len(yds), 3), device=dev)
        for i in range(3):
            Zds[:, :, i] = Z
            Yds[:, :, i] = Y
        pixel_coordinates = (
            self.det_corner_0.reshape(1, 1, 3)
            + Zds * self.zdhat.reshape(1, 1, 3)
            + Yds * self.ydhat.reshape(1, 1, 3)
        )
        return pixel_coordinates

    def _get_intersection(
        self, ray_direction: torch.Tensor, source_point: torch.Tensor
    ) -> torch.Tensor:
        """Get detector intersection coordinates for rays.

        Parameters
        ----------
        ray_direction : torch.Tensor
            Direction vectors for each ray, shape ``(N, 3)`` or ``(3,)``.
        source_point : torch.Tensor
            Origin points for each ray, shape ``(N, 3)`` or ``(3,)``.

        Returns
        -------
        torch.Tensor
            Intersection coordinates ``(zd, yd)`` and incident angles, shape
            ``(N, 3)``.
        """
        # Handle single vector inputs by adding batch dimension
        if ray_direction.dim() == 1:
            ray_direction = ray_direction.unsqueeze(0)
        if source_point.dim() == 1:
            source_point = source_point.unsqueeze(0)
        
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

    def _get_wrapping_cone(
        self, k: torch.Tensor, source_point: torch.Tensor
    ) -> torch.Tensor:
        """Compute cone that wraps detector corners around wavevector.

        Parameters
        ----------
        k : torch.Tensor
            Central wavevector of cone, shape ``(3,)``.
        source_point : torch.Tensor
            Cone vertex point, shape ``(3,)``.

        Returns
        -------
        torch.Tensor
            Half-angle of cone opening in radians.
        """
        # Ensure k and source_point are torch tensors
        k = ensure_torch(k)
        source_point = ensure_torch(source_point)
        
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

    def _detector_coordinate_to_pixel_index(
        self, zd: float, yd: float
    ) -> Tuple[int, int]:
        """Convert detector coordinates to pixel indices.

        Parameters
        ----------
        zd : float
            Z-axis detector coordinate in microns.
        yd : float
            Y-axis detector coordinate in microns.

        Returns
        -------
        tuple of int
            ``(row_index, col_index)`` pixel indices.
        """
        row_index = int(zd / self.pixel_size_z)
        col_index = int(yd / self.pixel_size_y)
        return row_index, col_index

    def _get_projected_bounding_box(
        self, proj_context: Dict[str, Any]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute bounding detector pixel indices for a convex hull projection.

        Parameters
        ----------
        proj_context : dict
            Dictionary containing:

            - ``'convex_hull'``: ConvexHull object to project.
            - ``'scattered_wave_vector'``: Direction of scattered wave.

        Returns
        -------
        tuple of int or None
            ``(min_row, max_row, min_col, max_col)`` indices for slicing the
            detector frame array. Returns ``None`` if the projection does not
            intersect the detector.
        """
        # Get vertices from convex hull (numpy) and convert to torch
        hull = proj_context['convex_hull']
        vertices = ensure_torch(hull.points[hull.vertices])

        # Reshape scattered_wave_vector to (N,3) by repeating for each vertex
        scattered_vec = proj_context['scattered_wave_vector']
        scattered_vec = scattered_vec.unsqueeze(0).repeat(vertices.shape[0], 1)
        
        # Get intersections - stays in torch
        projected_vertices = self._get_intersection(scattered_vec, vertices)

        # Calculate bounds using torch operations
        zmax = self.zmax
        ymax = self.ymax

        # Get z and y columns, filter out NaN values for min/max
        z_vals = projected_vertices[:, 0]
        y_vals = projected_vertices[:, 1]
        z_valid = z_vals[~torch.isnan(z_vals)]
        y_valid = y_vals[~torch.isnan(y_vals)]
        
        if z_valid.numel() == 0 or y_valid.numel() == 0:
            return None

        min_zd = max(float(z_valid.min().item()), 0)
        max_zd = min(float(z_valid.max().item()), float(zmax))
        min_yd = max(float(y_valid.min().item()), 0)
        max_yd = min(float(y_valid.max().item()), float(ymax))

        if min_zd > max_zd or min_yd > max_yd:
            return None

        min_row_indx, min_col_indx = self._detector_coordinate_to_pixel_index(
            min_zd, min_yd
        )
        max_row_indx, max_col_indx = self._detector_coordinate_to_pixel_index(
            max_zd, max_yd
        )

        max_row_indx = min(max_row_indx + 1, int(zmax / self.pixel_size_z))
        max_col_indx = min(max_col_indx + 1, int(ymax / self.pixel_size_y))

        return min_row_indx, max_row_indx, min_col_indx, max_col_indx

    # ------------------------------------------------------------------
    # 4. Peak / frame preprocessing
    # ------------------------------------------------------------------

    def _peaks_detector_intersection(
        self,
        peaks: torch.Tensor,
        columns: list[str],
        frames_to_render: int,
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Compute detector intersections, filter visible peaks, and assign frame indices.

        Also calculates and stores intensity factors (scattering strength per
        unit volume) based on structure, polarization, and Lorentz factors.

        Parameters
        ----------
        peaks : torch.Tensor
            Peaks tensor containing diffraction peak data.
        columns : list of str
            List of column names corresponding to the peaks tensor.
        frames_to_render : int
            Number of frames to render.

        Returns
        -------
        tuple
            ``(peaks, columns)`` where peaks is the filtered and augmented tensor
            and columns is the updated list of column names.
        """


        zd_yd_angle = self._get_intersection(peaks[:, 13:16], peaks[:, 16:19])
        peaks = torch.cat((peaks, zd_yd_angle), dim=1)
        columns.extend(["zd", "yd", "incident_angle"])

        # Filter peaks by detector bounds
        mask = self.contains(peaks[:, 25], peaks[:, 26])  # Shifted by 1 due to large_grain column
        peaks = peaks[mask]
        time = peaks[:, 6].contiguous()
        
        # Handle frame assignment: when frames_to_render==1, assign all to frame 0
        if frames_to_render <= 1:
            frame = torch.zeros((peaks.shape[0], 1), dtype=torch.int64, device=peaks.device)
        else:
            bins = torch.linspace(0, 1, steps=frames_to_render+1).contiguous()
            frame = torch.bucketize(time, bins).unsqueeze(1) - 1
            # Clamp frame indices to valid range [0, frames_to_render-1]
            frame = torch.clamp(frame, 0, frames_to_render - 1)
        
        peaks = torch.cat((peaks, frame), dim=1)
        columns.append("frame")

        # Compute intensity factors (scattering strength per unit volume)
        # This is the product of structure, polarization, and Lorentz factors
        intensity_factors = torch.ones(peaks.shape[0], device=peaks.device)
        if self.structure_factor:
            intensity_factors = intensity_factors * peaks[:, 5]
        if self.polarization_factor:
            intensity_factors = intensity_factors * peaks[:, 20]
        if self.lorentz_factor:
            intensity_factors = intensity_factors * peaks[:, 19]

        intensity_factors = intensity_factors.unsqueeze(1)
        peaks = torch.cat((peaks, intensity_factors), dim=1)
        columns.append("intensity_factors")

        # Filter out peaks with infinite or invalid intensity factors (geometrically impossible reflections)
        # These occur when Lorentz factor is infinite (eta near 0° or 180°, or theta near 0°)
        valid_intensity_mask = torch.isfinite(peaks[:, 29]) & (peaks[:, 29] > 0)
        n_invalid = (~valid_intensity_mask).sum().item()
        
        if n_invalid > 0 and self._verbose:
            import warnings
            # Count specific reasons for invalid peaks
            infinite_mask = torch.isinf(peaks[:, 29])
            n_infinite = infinite_mask.sum().item()
            n_negative_or_zero = ((peaks[:, 29] <= 0) & ~infinite_mask).sum().item()
            n_nan = torch.isnan(peaks[:, 29]).sum().item()
            
            reason_parts = []
            if n_infinite > 0:
                reason_parts.append(
                    f"{n_infinite} with infinite Lorentz factor (eta ≈ 0°/180° or θ ≈ 0°)"
                )
            if n_negative_or_zero > 0:
                reason_parts.append(f"{n_negative_or_zero} with non-positive intensity")
            if n_nan > 0:
                reason_parts.append(f"{n_nan} with NaN intensity")
            
            reason_str = "; ".join(reason_parts)
            warnings.warn(
                f"Skipping {n_invalid} peaks with invalid intensity factors: {reason_str}. "
                f"These represent geometrically impossible reflections.",
                UserWarning,
                stacklevel=3  # Points to render() caller
            )
        
        peaks = peaks[valid_intensity_mask]

        return peaks, columns

    # ------------------------------------------------------------------
    # 5. Rendering internals – grouped by method
    #    (order matches the 'method' options in render())
    # ------------------------------------------------------------------
    # 5.a Gaussian peaks
    def _render_gauss_peaks(self, peaks, frames_to_render, render_dtype: torch.dtype) -> torch.Tensor:
        """Render Gaussian peaks onto detector frames using optimized interpolation.

        This method uses σ=1.21 for Gaussian interpolation, which is equivalent to
        the combined effect of σ=0.7 interpolation + σ=0.99 convolution. This
        eliminates the need for a separate convolution step, providing 2.7×
        speedup on both CPU and GPU while producing identical results.

        Mathematical basis: ``G(σ₁) ⊗ G(σ₂) = G(√(σ₁² + σ₂²))``
        where ``√(0.7² + 0.99²) = 1.21``.

        Parameters
        ----------
        peaks : torch.Tensor
            Peak data tensor containing positions and intensities.
        frames_to_render : int
            Number of frames to generate.
        render_dtype : torch.dtype
            Data type for rendered frames.

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames with Gaussian peaks, shape
            ``(frames, height, width)``.

        Notes
        -----
        Peak tensor column mapping (after ``_peaks_detector_intersection``):

        - 0-24: Original columns from polycrystal (grain_index through peak_index)
        - 25: zd (detector x-coordinate in pixels)
        - 26: yd (detector y-coordinate in pixels)
        - 27: incident_angle (in degrees)
        - 28: frame (frame index for time-resolved rendering)
        - 29: intensity_factors (scattering strength per unit volume)
        """

        # Early exit for empty peak list
        if peaks.shape[0] == 0:
            return torch.zeros(
                (frames_to_render, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=self.det_corner_0.device,
                dtype=render_dtype,
            )

        diffraction_frames = torch.zeros(
            (
                frames_to_render,
                self.pixel_coordinates.shape[0],
                self.pixel_coordinates.shape[1],
            ),
            device=peaks.device,
            dtype=render_dtype,
        )

        # Get continuous pixel coordinates
        pos_z = peaks[:, 25] / self.pixel_size_z  # zd coordinate
        pos_y = peaks[:, 26] / self.pixel_size_y  # yd coordinate

        # Get integer coordinates for Gaussian interpolation
        z_center = pos_z.round().long()
        y_center = pos_y.round().long()
        
        # Compute total intensity = intensity_factors × volume
        intensity_factors = peaks[:, 29]  # Scattering strength per unit volume
        volumes = peaks[:, 21]  # Element volumes
        intensities = intensity_factors * volumes  # Total peak intensity

        cuda_selected = get_selected_device() == "cuda"

        # Gaussian interpolation parameters (OPTIMIZED for performance)
        # Combines interpolation σ_interp + detector PSF σ_detector into single step
        # Mathematical basis: G(σ₁) ⊗ G(σ₂) = G(√(σ₁² + σ₂²))
        # Using σ_interp=0.7 for sub-pixel interpolation
        sigma_interp = 0.7
        sigma = float((sigma_interp**2 + self.gaussian_sigma**2) ** 0.5)
        radius = max(3, int(3 * sigma) + 1)  # ≈3σ coverage for >99% energy
        
        # Pre-compute Gaussian neighborhood offsets
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float64, device=pos_z.device)
        dz_grid, dy_grid = torch.meshgrid(offsets, offsets, indexing='ij')
        
        # Flatten grids for easier processing
        dz_flat = dz_grid.flatten().to(torch.int32)  # Shape: (25,)
        dy_flat = dy_grid.flatten().to(torch.int32)  # Shape: (25,)
        
        # Batch peaks to limit intermediate memory usage. Estimate a batch size based
        # on free VRAM when CUDA is the selected device; otherwise fall back to a default.
        approx_bytes_per_peak = 25 * (4 + 4 + 4 + 4 + 4)  # rough int/float32 intermediates per offset

        try:
            free_bytes = return_device_memory().get("free_gb")
            target_bytes = free_bytes * 0.5 if free_bytes > 0 else 0
            batch_size = int(target_bytes / max(approx_bytes_per_peak, 1))
            batch_size = max(10_000, min(batch_size, 200_000)) if batch_size > 0 else 50_000
        except Exception:
            batch_size = 50_000

        n_peaks = pos_z.shape[0]
        n_batches = (n_peaks + batch_size - 1) // batch_size if batch_size > 0 else 1
        for batch_idx, start in enumerate(range(0, n_peaks, batch_size)):
            end = min(start + batch_size, n_peaks)
            if self._verbose:
                progress_fraction = end / n_peaks
                utils._print_progress(
                    progress_fraction,
                    f"[Micro] Batch {batch_idx+1}/{n_batches}"
                )

            z_centers_exp = z_center[start:end].to(torch.int32).unsqueeze(1)  
            pos_z_exp = pos_z[start:end].float().unsqueeze(1)                 
            z_pixels = z_centers_exp + dz_flat.unsqueeze(0)  
            z_dist = z_pixels.float() - pos_z_exp          
            del z_centers_exp, pos_z_exp

            y_centers_exp = y_center[start:end].to(torch.int32).unsqueeze(1)  
            pos_y_exp = pos_y[start:end].float().unsqueeze(1)                 
            y_pixels = y_centers_exp + dy_flat.unsqueeze(0)  
            y_dist = y_pixels.float() - pos_y_exp          
            del y_centers_exp, pos_y_exp

            dist_sq = z_dist**2 + y_dist**2
            
            weights = torch.exp(-dist_sq / (2 * sigma**2))  
            weight_sums = weights.sum(dim=1, keepdim=True)
            weights = weights / (weight_sums + 1e-12)
            weights = weights * intensities[start:end].unsqueeze(1)
            
            valid = (z_pixels >= 0) & (z_pixels < diffraction_frames.shape[1]) & \
                   (y_pixels >= 0) & (y_pixels < diffraction_frames.shape[2]) & \
                   (weights > 1e-6)  # Only significant weights
            
            if valid.any():
                valid_peaks, valid_offsets = valid.nonzero(as_tuple=True)
                z_coords = z_pixels[valid].long()  # Convert to long for indexing
                y_coords = y_pixels[valid].long()  # Convert to long for indexing
                valid_weights = weights[valid]
                frame_indices = peaks[start:end][valid_peaks, 28].long()  # Frame indices for this batch
                
                diffraction_frames.index_put_(
                    (frame_indices, z_coords, y_coords),
                    valid_weights.to(render_dtype),
                    accumulate=True
                )

        return diffraction_frames

    # 5.b Airy peaks
    def _estimate_airy_kernel_memory(
        self, 
        fwhm_rad: torch.Tensor, 
        incident_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate memory requirement per peak based on Airy kernel size.

        Memory scales quadratically with kernel radius. Smaller crystallites have
        larger FWHM (Scherrer broadening), leading to larger kernels and more
        memory usage.

        The kernel radius is determined by:

        - Airy: ``first_zero = FWHM_rad * R / pixel_size * 1.1861``
        - Final radius is approximately 4-5× the characteristic width until
          intensity drops below threshold.

        Parameters
        ----------
        fwhm_rad : torch.Tensor
            Scherrer FWHM in radians, shape ``(N,)``.
        incident_angles : torch.Tensor
            Incident angles in degrees, shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Estimated memory per peak in bytes, shape ``(N,)``.
        """
        detector_distance = torch.linalg.norm(self.det_corner_0)
        incident_angles_rad = incident_angles * torch.pi / 180
        R = detector_distance / torch.cos(incident_angles_rad)
        
        # Compute characteristic width parameter in pixels
        # Airy: first_zero = FWHM * R / pixel_size * 1.1861
        char_width = fwhm_rad * R / self.pixel_size_z * 1.1861
        # Airy needs ~4× first_zero to capture rings
        kernel_radius = 4.0 * char_width
        
        # Ensure minimum radius (Gaussian PSF)
        min_radius = 3.0 * self.gaussian_sigma
        kernel_radius = torch.maximum(kernel_radius, torch.tensor(min_radius))
        
        # Kernel size = 2 * radius + 1
        kernel_size = 2 * kernel_radius.ceil() + 1
        
        # Memory = kernel_size² × 8 bytes (float64) × overhead factor (4× for buffers)
        memory_bytes = kernel_size ** 2 * 8.0 * 4.0
        
        return memory_bytes
    
    def _render_airy_peaks(
        self,
        peaks,
        frames_to_render,
        render_dtype: torch.dtype,
        memory_safety_factor: float = 3.0,
    ) -> torch.Tensor:
        """Render Airy disk peaks with memory-aware batching.

        Parameters
        ----------
        peaks : torch.Tensor
            Processed peaks tensor with detector intersections and intensity
            factors.
        frames_to_render : int
            Number of frames to generate.
        render_dtype : torch.dtype
            Data type for rendered frames.
        memory_safety_factor : float, optional
            Multiplier for memory estimation to ensure safe batching.
            Default is 3.0.

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames with broadened peaks, shape
            ``(frames, height, width)``.
        """
        # Early exit for empty peak list
        if peaks.shape[0] == 0:
            return torch.zeros(
                (frames_to_render, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=self.det_corner_0.device,
                dtype=render_dtype,
            )

        # 1. Estimate memory per peak based on kernel fwhm_rad
        memory_per_peak = self._estimate_airy_kernel_memory(
            peaks[:, 23], peaks[:, 27]
        ) * memory_safety_factor

        # 2. Estimate available memory (works for both CPU and GPU)
        available_bytes = return_device_memory().get("free_gb", 1.0) * (1024**3)

        # 3. Sort peaks by memory (largest first)
        sort_indices = torch.argsort(memory_per_peak, descending=True)
        peaks = peaks[sort_indices]
        memory_per_peak = memory_per_peak[sort_indices]

        # Initialize output frames
        frames = torch.zeros(
            (frames_to_render, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
            device=peaks.device,
            dtype=render_dtype,
        )

        # 4-6. Process in batches that fit in available memory
        n_peaks = peaks.shape[0]
        start_idx = 0
        batch_num = 0

        while start_idx < n_peaks:
            # Find how many peaks fit in available memory
            cumulative_memory = torch.cumsum(memory_per_peak[start_idx:], dim=0)
            fits_mask = cumulative_memory <= available_bytes
            batch_size = max(1, int(fits_mask.sum().item()))

            end_idx = start_idx + batch_size
            batch_peaks = peaks[start_idx:end_idx]

            # Extract batch data
            fwhm_rad = batch_peaks[:, 23]
            incident_angles = batch_peaks[:, 27]
            zd = batch_peaks[:, 25] / self.pixel_size_z
            yd = batch_peaks[:, 26] / self.pixel_size_y
            intensities = batch_peaks[:, 29] * batch_peaks[:, 21]

            # Generate Airy disk kernels
            kernels = self._airy_kernel_batch(fwhm_rad, incident_angles)
            kernels = (kernels * intensities.view(-1, 1, 1, 1)).to(render_dtype)

            # Deposit kernels to appropriate frames
            for frame_idx in range(frames_to_render):
                frame_mask = batch_peaks[:, 28] == frame_idx
                if frame_mask.any():
                    frames[frame_idx] = self._deposit_kernels_batch(
                        frames[frame_idx],
                        kernels[frame_mask],
                        zd[frame_mask],
                        yd[frame_mask],
                        render_dtype,
                    )

            if self._verbose:
                progress = end_idx / n_peaks
                utils._print_progress(progress, f"[Nano] Batch {batch_num + 1}, n={batch_size}")

            del kernels
            start_idx = end_idx
            batch_num += 1

        # Apply Gaussian convolution for detector point spread function
        for i in range(frames_to_render):
            frames[i] = self._conv2d_gaussian_kernel(frames[i])

        return frames

    def _airy_kernel_batch(
        self, fwhm_rad: torch.Tensor, incident_angles: torch.Tensor
    ) -> torch.Tensor:
        """Generate multiple 2D Airy disk kernels for crystallite diffraction.

        The Airy disk is the physically correct diffraction pattern from a
        circular aperture (crystallite). The pattern width scales inversely with
        crystallite size:

        - Small crystallites → broad Airy patterns
        - Large crystallites → narrow Airy patterns (approaching delta function)

        The Airy intensity pattern is: ``I(x) = I_0 * [2*J_1(x)/x]^2``
        where ``x = 3.8317 * r / first_zero_radius`` (first zero of J_1 is at
        3.8317).

        The FWHM of the Airy pattern is approximately ``0.847 * first_zero_radius``.
        Therefore: ``first_zero_radius = FWHM / 0.847 ≈ FWHM * 1.181``.

        This ensures that when the Scherrer FWHM is given, the resulting Airy
        disk has matching FWHM, allowing accurate crystallite size recovery.

        Parameters
        ----------
        fwhm_rad : torch.Tensor
            FWHM values in radians from Scherrer broadening, shape ``(N,)``.
        incident_angles : torch.Tensor
            Incident angles in degrees for each peak, shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Batch of normalized Airy disk kernels with shape ``(N, 1, H, W)``.
        """
        detector_distance = torch.linalg.norm(self.det_corner_0)
        incident_angles_rad = incident_angles * torch.pi / 180
        R = detector_distance / torch.cos(incident_angles_rad)

        # Convert FWHM in radians to pixel scale
        fwhm_pixels = (fwhm_rad * R / self.pixel_size_z)
        
        # The Airy pattern has FWHM ≈ 0.843 * first_zero_radius
        # Therefore: first_zero_radius = FWHM / 0.843 ≈ FWHM * 1.1861
        # This ensures the rendered Airy disk has the correct FWHM for Scherrer recovery
        AIRY_FWHM_TO_FIRST_ZERO = 1.1861  # = 1 / 0.843
        first_zero_pixels = fwhm_pixels * AIRY_FWHM_TO_FIRST_ZERO

        sigma = self.gaussian_sigma
        max_first_zero = first_zero_pixels.max()
        threshold = self.kernel_threshold / 2

        # Calculate kernel radius - need to capture several Airy rings
        # Airy pattern decays as 1/x^3 in the tails, so we need larger extent
        # for small crystallites (broad patterns)
        # Use ~4 times the first zero to capture sufficient rings
        radius = torch.tensor(max(4.0 * float(max_first_zero), 8.0))

        # Ensure we also capture Gaussian tails (cap at 128 pixels)
        max_airy_radius = 128
        while radius <= max_airy_radius:
            g_val = torch.exp(-(radius**2) / (2 * sigma**2))
            if g_val < threshold:
                break
            radius = radius * 2
        if radius > max_airy_radius:
            if self._verbose:
                print(f"Airy kernel radius should be {radius.item()} but limited to {max_airy_radius}")
            radius = torch.tensor(max_airy_radius)

        ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
        yy, zz = torch.meshgrid(ax, ax, indexing="ij")
        r = torch.sqrt(yy**2 + zz**2)

        kernels = []
        max_size = 0

        # First zero of J_1(x) occurs at x ≈ 3.8317
        FIRST_ZERO_J1 = 3.8317

        def airy_profile(r: npt.NDArray, first_zero: float, sigma: float) -> npt.NDArray:
            """Compute Airy disk profile convolved with Gaussian PSF.

            The Airy intensity pattern is: ``I(x) = [2*J_1(x)/x]^2``
            where ``x = 3.8317 * r / first_zero_radius``.

            The first zero of J_1(x) is at x = 3.8317, so the pattern's
            first zero occurs at r = first_zero_radius.

            For r=0, the pattern has value 1 (use L'Hôpital's rule).

            Parameters
            ----------
            r : numpy.ndarray
                Radial distances in pixels.
            first_zero : float
                First zero radius of Airy pattern in pixels.
            sigma : float
                Gaussian width for instrument broadening within the Airy kernel.

            Returns
            -------
            numpy.ndarray
                Computed Airy disk values (normalized later).
            """
            # Scale r so that first zero occurs at r = first_zero
            # x = 3.8317 * r / first_zero, so at r = first_zero, x = 3.8317
            x = np.where(r == 0, 1e-10, FIRST_ZERO_J1 * r / first_zero)
            
            # Airy pattern: [2*J_1(x)/x]^2
            # Handle small x separately to avoid numerical issues
            airy = np.where(
                np.abs(x) < 1e-8,
                1.0,  # At center, the jinc function approaches 1
                (2 * j1(x) / x) ** 2
            )
            
            # For very small crystallites (broad patterns), the Airy disk dominates
            # For large crystallites (narrow patterns), convolve with Gaussian PSF
            if first_zero < sigma:
                # Pattern is sharper than PSF - convolve with Gaussian
                # Simple approximation: multiply by Gaussian to smooth
                gaussian = np.exp(-(r**2) / (2 * sigma**2))
                return airy * gaussian
            else:
                return airy

        r_np = r.cpu().numpy()
        sigma_np = float(sigma)

        for first_zero in first_zero_pixels:
            first_zero_np = float(first_zero)
            
            # For very large crystallites (very small first_zero), use Gaussian kernel
            if first_zero_np < 0.5:
                kernel = self.gaussian_kernel
            else:
                kernel = ensure_torch(airy_profile(r_np, first_zero_np, sigma_np))
            
            kernel = kernel / kernel.sum()

            # Crop to significant region using binary search
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

            crop_r = left
            cropped = kernel[center - crop_r : center + crop_r + 1, center - crop_r : center + crop_r + 1]

            kernels.append(cropped.unsqueeze(0).unsqueeze(0))
            max_size = max(max_size, 2 * crop_r + 1)

        # Pad all kernels to same size
        padded_kernels = []
        for k in kernels:
            if k.shape[-1] < max_size:
                pad = (max_size - k.shape[-1]) // 2
                k_padded = F.pad(k, (pad, pad, pad, pad), mode="constant", value=0)
                padded_kernels.append(k_padded.to(device=first_zero_pixels.device))
            else:
                padded_kernels.append(k.to(device=first_zero_pixels.device))

        result = torch.cat(padded_kernels, dim=0)
        return result

    def _deposit_kernels_batch(
        self,
        tensor: torch.Tensor,
        kernels: torch.Tensor,
        centers_z: torch.Tensor,
        centers_y: torch.Tensor,
        render_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Deposit multiple kernels onto a tensor with sub-pixel precision.

        Uses bilinear interpolation (4 neighbors) for sub-pixel positioning,
        which is memory-efficient while preserving positional accuracy.

        Parameters
        ----------
        tensor : torch.Tensor
            Target tensor to deposit onto, shape ``(H, W)``.
        kernels : torch.Tensor
            Batch of kernels to deposit, shape ``(N, 1, H, W)``.
        centers_z : torch.Tensor
            Z coordinates for kernel centers (in pixels, sub-pixel precision).
        centers_y : torch.Tensor
            Y coordinates for kernel centers (in pixels, sub-pixel precision).
        render_dtype : torch.dtype
            Data type for output values.

        Returns
        -------
        torch.Tensor
            Updated tensor with deposited kernels.
        """
        N = kernels.shape[0]
        kh, kw = kernels.shape[-2:]
        half_h, half_w = kh // 2, kw // 2
        H, W = tensor.shape

        # Create kernel pixel offsets (relative to center)
        offsets_z = torch.arange(kh, device=kernels.device, dtype=torch.float64) - half_h
        offsets_y = torch.arange(kw, device=kernels.device, dtype=torch.float64) - half_w

        # Build all sub-pixel positions: (N, kh, kw)
        all_z = centers_z.view(N, 1, 1) + offsets_z.view(1, kh, 1)
        all_y = centers_y.view(N, 1, 1) + offsets_y.view(1, 1, kw)
        
        # Expand to full grid and flatten
        all_z = all_z.expand(N, kh, kw).reshape(-1)
        all_y = all_y.expand(N, kh, kw).reshape(-1)
        
        # Flatten kernel values
        values = kernels.view(N, -1).reshape(-1)

        # Bilinear interpolation: split into integer and fractional parts
        z0 = all_z.floor().long()
        y0 = all_y.floor().long()
        z1 = z0 + 1
        y1 = y0 + 1
        
        # Fractional parts for weighting
        fz = all_z - z0.float()
        fy = all_y - y0.float()
        
        # Bilinear weights for 4 corners
        w00 = (1 - fz) * (1 - fy)  # top-left
        w01 = (1 - fz) * fy        # top-right
        w10 = fz * (1 - fy)        # bottom-left
        w11 = fz * fy              # bottom-right

        # Deposit to all 4 corners with their weights
        for zi, yi, w in [(z0, y0, w00), (z0, y1, w01), (z1, y0, w10), (z1, y1, w11)]:
            valid = (zi >= 0) & (zi < H) & (yi >= 0) & (yi < W)
            if valid.any():
                tensor.index_put_(
                    (zi[valid], yi[valid]),
                    (values[valid] * w[valid]).to(render_dtype),
                    accumulate=True,
                )

        return tensor

    # 5.c Projected volumes
    def _render_projected_volumes(self, peaks, beam, mesh_lab, rigid_body_motion, frames_to_render, render_dtype: torch.dtype) -> torch.Tensor:
        """Render projected volumes onto detector frames using volume projection.

        This provides accurate peak shapes by projecting 3D crystal volumes.

        Parameters
        ----------
        peaks : torch.Tensor
            Processed peaks tensor with detector intersections and intensity
            factors.
        beam : Beam
            Beam object for computing convex hull intersections.
        mesh_lab : Mesh
            Mesh object containing tetrahedral element geometry.
        rigid_body_motion : RigidBodyMotion
            RigidBodyMotion object for transforming vertices.
        frames_to_render : int
            Number of frames to generate.
        render_dtype : torch.dtype
            Data type for rendered frames.

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames with projected volumes, shape
            ``(frames, height, width)``.
        """
        frames_bundle = peaks[:, 28].unique()  # frame column is index 28

                # Early exit for empty peak list
        if peaks.shape[0] == 0:
            return torch.zeros(
                (frames_to_render, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=self.det_corner_0.device,
                dtype=render_dtype,
            )

        device = peaks.device
        scattered_vectors = peaks[:, 13:16]
        
        # Get element indices from peaks (column 0: grain_index which maps to element)
        element_indices = peaks[:, 0].long()
        times = peaks[:, 6]
        # Initialize all frames to zeros so we can write directly by frame index
        diffraction_frames = torch.zeros((frames_to_render, self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]), device=device, dtype=render_dtype)
        
        # Get vertices for each peak and apply rigid body motion
        node_indices = mesh_lab.enod[element_indices]
        vertices = ensure_torch(mesh_lab.coord[node_indices])  # Shape: (N_peaks, 4, 3)
        
        # Apply rigid body motion to all vertices at their corresponding times
        # RigidBodyMotion handles (N, 4, 3) shape with per-vector times (N,)
        rotated_vertices = rigid_body_motion(vertices, times)  # Shape: (N_peaks, 4, 3)
        
        total_peaks = peaks.shape[0]
        processed_peaks = 0

        for frame_index in frames_bundle:
            # Initialize frame on same device as peaks
            frames = torch.zeros(
                (self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=device,
                dtype=render_dtype,
            )
            
            # Process each peak in this frame
            frame_mask = peaks[:, 28] == frame_index
            frame_peak_indices = torch.where(frame_mask)[0]
            
            for i, peak_idx in enumerate(frame_peak_indices):
                processed_peaks += 1
                # Show progress for verbose mode using a single continuous bar
                if self._verbose and total_peaks > 0:
                    progress_fraction = processed_peaks / total_peaks
                    utils._print_progress(
                        progress_fraction,
                        f"[Macro] Peak {processed_peaks}/{total_peaks}"
                    )
                
                # Compute convex hull intersection with beam
                hull = beam._intersect(rotated_vertices[peak_idx])
                
                if hull is not None:
                    # Create minimal projection context
                    proj_context = {
                        'convex_hull': hull,
                        'scattered_wave_vector': scattered_vectors[peak_idx]
                    }
                    
                    # Project volume using only necessary data
                    box = self._get_projected_bounding_box(proj_context)
                    if box is not None:
                        # Keep projection in numpy for computation
                        projection = self._project_convex_hull(proj_context, box)
                        # Use pre-calculated intensity from peaks tensor
                        intensity = peaks[peak_idx, 29]  # intensity column is index 29                        
                        
                        # Note: Infinite intensity peaks should already be filtered out
                        # by _peaks_detector_intersection. This check is defensive.
                        if torch.isfinite(intensity):
                            projection_torch = ensure_torch(projection)
                            # projection is path length, multiply by pixel area to get volume
                            projected_volume = projection_torch * self.pixel_size_z * self.pixel_size_y
                            # Multiply intensity per volume by projected volume
                            result = (intensity * projected_volume).to(render_dtype)
                            frames[box[0]:box[1], box[2]:box[3]] += result
            
            # Apply Gaussian convolution for point spread function
            frames = self._conv2d_gaussian_kernel(frames)
            diffraction_frames[frame_index.long()] = frames
            
        return diffraction_frames

    def _project_convex_hull(
        self, proj_context: Dict[str, Any], box: Tuple[int, int, int, int]
    ) -> npt.NDArray:
        """Project convex hull onto detector region.

        Parameters
        ----------
        proj_context : dict
            Dictionary containing:

            - ``'convex_hull'``: ConvexHull object to project.
            - ``'scattered_wave_vector'``: Direction of scattered wave.

        box : tuple of int
            ``(min_z, max_z, min_y, max_y)`` bounds for projection region.

        Returns
        -------
        torch.Tensor
            Array of clip lengths (path lengths) through the hull for each
            detector ray.
        """
        # Get pixel coordinates - keep in torch
        pixel_coords = self.pixel_coordinates[box[0]:box[1], box[2]:box[3], :]
        ray_points = pixel_coords.reshape(
            (box[1] - box[0]) * (box[3] - box[2]), 3
        )

        # ConvexHull from scipy returns numpy, convert to torch
        hull = proj_context['convex_hull']
        plane_normals = ensure_torch(hull.equations[:, 0:3])
        plane_offsets = ensure_torch(hull.equations[:, 3]).reshape(
            hull.equations.shape[0], 1
        )
        plane_points = -plane_offsets * plane_normals

        # Ray direction in torch
        scattered_vec = proj_context['scattered_wave_vector']
        ray_direction = scattered_vec / torch.linalg.norm(scattered_vec)

        clip_lengths = utils._clip_line_with_convex_polyhedron(
            ray_points, ray_direction, plane_points, plane_normals
        )
        clip_lengths = clip_lengths.reshape(box[1] - box[0], box[3] - box[2])

        return clip_lengths

    # ------------------------------------------------------------------
    # 6. Convolution / kernel helpers
    # ------------------------------------------------------------------
    def _conv2d_gaussian_kernel(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply the point spread function to detector frames using Gaussian convolution.

        Parameters
        ----------
        frames : torch.Tensor
            Input frames to convolve, shape ``(H, W)``, ``(N, H, W)``, or
            ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Convolved frames with same shape as input.
        """
        frames = ensure_torch(frames)

        input_dims = frames.ndim
        if input_dims == 2:
            frames = frames.unsqueeze(0).unsqueeze(0)
        elif input_dims == 3:
            frames = frames.unsqueeze(1)

        kernel = self.gaussian_kernel.to(frames.dtype).unsqueeze(0).unsqueeze(0)
        kernel_padding = kernel.shape[-1] // 2

        with torch.no_grad():
            output = torch.nn.functional.conv2d(frames, weight=kernel, padding=kernel_padding)

        if input_dims == 2:
            output = output.squeeze(0).squeeze(0)
        elif input_dims == 3:
            output = output.squeeze(1)

        return output

    def _generate_gaussian_kernel(self) -> torch.Tensor:
        """Generate a normalized 2D Gaussian kernel with dynamic size.

        The kernel size is determined by the ``kernel_threshold`` attribute,
        expanding until the Gaussian tail drops below the threshold.

        Returns
        -------
        torch.Tensor
            2D Gaussian kernel of shape ``(H, W)``, normalized to sum to 1.
        """
        # Get device from det_corner_0 to ensure kernel is on same device
        dev = self.det_corner_0.device
        
        sigma = ensure_torch(self.gaussian_sigma).to(dev)
        threshold = self.kernel_threshold

        # DEFAULT: Limit Gaussian kernel to maximum 5x5 for consistency with Airy method
        # This ensures both Gauss and Airy use similar kernel sizes, providing
        # excellent agreement (<1% error) in azimuthally integrated profiles
        max_radius = 2  # gives 5x5 kernel (2*2+1=5)

        radius = 1
        while radius <= max_radius:
            value = torch.exp(-(radius**2) / (2 * sigma**2))
            if value < threshold / 2:
                break
            radius += 1
        radius = min(radius, max_radius)
        
        kernel_size = 2 * radius + 1

        ax = torch.arange(kernel_size, dtype=torch.float64, device=dev) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel  # Return as (H,W) instead of (1,1,H,W)


    # ------------------------------------------------------------------
    # 7. Deprecated methods (clearly separated)
    # ------------------------------------------------------------------
    # DEPRECATED METHODS - TO BE REMOVED IN FUTURE VERSION

    def _ray_detector_intersection(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """Find where a ray intersects the detector plane.

        .. deprecated::
            This method is deprecated and will be removed in a future version.

        Parameters
        ----------
        origin : numpy.ndarray
            Ray origin point, shape ``(3,)``.
        direction : numpy.ndarray
            Normalized ray direction, shape ``(3,)``.

        Returns
        -------
        numpy.ndarray or None
            3D intersection point, or ``None`` if no intersection exists.
        """
        # Detector plane equation: dot(point - det_corner_0, normal) = 0
        # Ray equation: point = origin + t * direction
        
        normal = ensure_numpy(self.normal)
        corner = ensure_numpy(self.det_corner_0)
        
        denom = np.dot(direction, normal)
        
        if abs(denom) < 1e-10:
            return None  # Ray parallel to plane
        
        t = np.dot(corner - origin, normal) / denom
        
        if t < 0:
            return None  # Intersection behind ray origin
        
        intersection = origin + t * direction
        return intersection

    def _world_to_pixel_coords(self, world_point: np.ndarray) -> Optional[Tuple[float, float]]:
        """Convert 3D world coordinates to 2D pixel coordinates.

        .. deprecated::
            This method is deprecated and will be removed in a future version.

        Parameters
        ----------
        world_point : numpy.ndarray
            3D point in lab frame, shape ``(3,)``.

        Returns
        -------
        tuple of float or None
            ``(z_pixel, y_pixel)`` coordinates, or ``None`` if outside detector.
        """
        corner = ensure_numpy(self.det_corner_0)
        zdhat = ensure_numpy(self.zdhat)
        ydhat = ensure_numpy(self.ydhat)
        
        # Vector from detector corner to point
        v = world_point - corner
        
        # Project onto detector axes
        z_coord = np.dot(v, zdhat)
        y_coord = np.dot(v, ydhat)
        
        # Convert to pixel coordinates
        z_pixel = z_coord / float(self.pixel_size_z)
        y_pixel = y_coord / float(self.pixel_size_y)
        
        # Check bounds
        if (z_pixel < 0 or z_pixel >= self.pixel_coordinates.shape[0] or
            y_pixel < 0 or y_pixel >= self.pixel_coordinates.shape[1]):
            return None
        
        return (z_pixel, y_pixel)

    def _compute_pixel_path_length(
        self, ray_point: np.ndarray, ray_direction: np.ndarray, hull
    ) -> float:
        """Compute exact path length of ray through convex hull.

        .. deprecated::
            This method is deprecated and will be removed in a future version.

        Parameters
        ----------
        ray_point : numpy.ndarray
            3D starting point of ray (pixel position).
        ray_direction : numpy.ndarray
            Normalized ray direction (scattered wave direction).
        hull : ConvexHull
            ConvexHull object from beam intersection.

        Returns
        -------
        float
            Path length through hull in microns.
        """
        # Use existing utility function for ray-polyhedron intersection
        from scipy.spatial import ConvexHull as SciPyConvexHull
        
        # Get hull faces (planes)
        hull_points = ensure_numpy(hull.points)
        try:
            scipy_hull = SciPyConvexHull(hull_points)
        except:
            return 0.0
        
        # Get plane normals and points
        plane_normals = scipy_hull.equations[:, :3]  # (n_faces, 3)
        plane_offsets = scipy_hull.equations[:, 3]   # (n_faces,)
        
        # Plane points (any point on each plane)
        plane_points = hull_points[scipy_hull.simplices[:, 0]]  # (n_faces, 3)
        
        # Convert to torch for utility function
        ray_point_torch = ensure_torch(ray_point).unsqueeze(0)  # (1, 3)
        ray_direction_torch = ensure_torch(ray_direction).unsqueeze(0)  # (1, 3)
        plane_points_torch = ensure_torch(plane_points)
        plane_normals_torch = ensure_torch(plane_normals)
        
        # Compute clip length
        clip_lengths = utils._clip_line_with_convex_polyhedron(
            ray_point_torch,
            ray_direction_torch,
            plane_points_torch,
            plane_normals_torch
        )
        
        return float(clip_lengths[0])


