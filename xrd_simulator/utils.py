"""
General internal package utility functions.

This module provides various utility functions used for internal package operations.
The functions include mathematical computations, geometric transformations, file handling,
and progress tracking.

Functions:
    _diffractogram: Compute diffractogram from pixelated diffraction pattern.
    _contained_by_intervals: Check if a value is contained within specified intervals.
    _cif_open: Open a CIF file using the ReadCif function from the CifFile module.
    _print_progress: Print a progress bar in the executing shell terminal.
    _clip_line_with_convex_polyhedron: Compute lengths of parallel lines clipped by a convex polyhedron.
    alpha_to_quarternion: Generate a unit quaternion from spherical angle coordinates on the S3 ball.
    lab_strain_to_B_matrix: Convert strain tensors in lab coordinates to lattice matrices (B matrices).
    _get_circumscribed_sphere_centroid: Compute the centroid of a circumscribed sphere for a given set of points.
    _get_bounding_ball: Compute a minimal bounding ball for a set of Euclidean points.
    _set_xfab_logging: Enable or disable logging for the xfab module.
    _verbose_manager: Manage global verbose options for logging within with statements.
    _strain_as_tensor: Convert a strain vector to a strain tensor.
    _strain_as_vector: Convert a strain tensor to a strain vector.
    _b_to_epsilon: Compute strain tensor from B matrix for large deformations.
    _epsilon_to_b: Compute B matrix from strain tensor for large deformations.
    _get_misorientations: Compute minimal angles required to rotate SO3 elements to their mean orientation.
    _compute_sides: Compute the lengths of the sides of tetrahedrons.
    _circumsphere_of_segments: Compute the minimum circumsphere of line segments.
    _circumsphere_of_triangles: Compute the minimum circumsphere of triangles.
    _circumsphere_of_tetrahedrons: Compute the circumcenter of tetrahedrons.
"""

from __future__ import annotations

import os
import logging
from itertools import combinations
import gc
from typing import Dict, List
from CifFile import ReadCif
from scipy.spatial.transform import Rotation
import numpy as np
import sys
import logging as _logging
import pandas as pd
import torch

from xrd_simulator.cuda import get_selected_device

import numpy as np
import pandas as pd
import torch


def peaks_dict_to_dataframe(peaks_dict) -> pd.DataFrame:
    """
    Convert a peaks dictionary (from Polycrystal.diffract) into a DataFrame.
    Handles torch tensors, numpy arrays, and fixes mismatched column lists.
    """

    # --- Validate structure ---
    if not isinstance(peaks_dict, dict):
        raise ValueError("peaks_dict must be a dictionary")

    try:
        peaks = peaks_dict["peaks"]
        columns = list(peaks_dict.get("columns", []))
    except KeyError:
        raise ValueError("peaks_dict must contain 'peaks' and 'columns' keys")

    # --- Convert peaks to NumPy ---
    if isinstance(peaks, torch.Tensor):
        peaks_np = peaks.detach().cpu().numpy()
    else:
        peaks_np = np.asarray(peaks)

    # Ensure 2D
    if peaks_np.ndim == 1:
        peaks_np = peaks_np.reshape(1, -1)

    ncols = peaks_np.shape[1]

    # --- Fix column count automatically ---
    if len(columns) != ncols:
        columns = (columns[:ncols] +           # truncate too-long list
                   [f"col_{i}" for i in range(len(columns), ncols)])  # pad if too short

    return pd.DataFrame(peaks_np, columns=columns)


def _diffractogram(diffraction_pattern, det_centre_z, det_centre_y, binsize=1.0):
    """Compute diffractogram from pixelated diffraction pattern.

    Args:
        diffraction_pattern (:obj:`numpy array`): Pixelated diffraction pattern``shape=(m,n)``
        det_centre_z (:obj:`numpy array`): Intersection pixel coordinate between
                 beam centroid line and detector along z-axis.
        det_centre_y (:obj:`list` of :obj:`float`): Intersection pixel coordinate between
                 beam centroid line and detector along y-axis.
        binsize  (:obj:`list` of :obj:`float`): Histogram binsize. (Detector pixels are integrated
            radially around the azimuth)

    Returns:
        (:obj:`tuple`) with ``bin_centres`` and ``histogram``.

    """
    m, n = diffraction_pattern.shape
    max_radius = np.max([m, n])
    bin_centres = np.arange(0, int(max_radius + 1), binsize)
    histogram = np.zeros((len(bin_centres),))
    for i in range(m):
        for j in range(n):
            radius = np.sqrt((i - det_centre_z) ** 2 + (j - det_centre_y) ** 2)
            bin_index = np.argmin(np.abs(bin_centres - radius))
            histogram[bin_index] += diffraction_pattern[i, j]
    clip_index = len(histogram) - 1
    for k in range(len(histogram) - 1, 0, -1):
        if histogram[k] != 0:
            break
        else:
            clip_index = k
    clip_index = np.min([clip_index + m // 10, len(histogram) - 1])
    return bin_centres[0:clip_index], histogram[0:clip_index]


def _contained_by_intervals(value, intervals):
    """Assert if a float ``value`` is contained by any of a number of ``intervals``."""
    for bracket in intervals:
        if value >= bracket[0] and value <= bracket[1]:
            return True
    return False


def _cif_open(cif_file):
    """Helper function to be able to use the ``.CIFread`` of ``xfab``."""
    cif_dict = ReadCif(cif_file)
    return cif_dict[list(cif_dict.keys())[0]]


def _print_progress(progress_fraction, message):
    """Print a progress bar in the executing shell terminal.

    Args:
        progress_fraction (:obj:`float`): progress between 0 and 1.
        message (:obj:`str`): Optional message prepend the loading bar with. (max 55 characters)

    """
    assert (
        len(message) <= 55.0
    ), "Message to print is too long, max 55 characters allowed."
    progress_in_precent = np.round(100 * progress_fraction, 1)
    progress_bar_length = int(progress_fraction * 40)
    print(
        "\r{0}{1} |{2}{3}|".format(
            message,
            " " * (55 - len(message)),
            "█" * progress_bar_length,
            " " * (40 - progress_bar_length),
        )
        + " "
        + str(progress_in_precent)
        + "%",
        end="",
    )
    if progress_fraction == 1.0:
        print("")

def _clip_line_with_convex_polyhedron(
    line_points, line_direction, plane_points, plane_normals
):
    """Torch-native vectorized clipping for many parallel lines.

    Args:
        line_points: Base points of rays (exterior to polyhedron), shape=(n,3)
        line_direction: Normalized ray direction (all rays have same direction), shape=(3,)
        plane_points: Point in each polyhedron face plane, shape=(m,3)
        plane_normals: Outwards element face normals, shape=(m,3)

    Returns:
        torch.Tensor: Intersection lengths, shape=(n,) on same device as line_points
    """
    # Convert inputs to torch (ensure_torch handles already-tensor inputs efficiently)
    line_points = ensure_torch(line_points)
    line_direction = ensure_torch(line_direction)
    plane_points = ensure_torch(plane_points)
    plane_normals = ensure_torch(plane_normals)

    t2 = torch.matmul(plane_normals, line_direction)  # (m,)
    te_mask = t2 < 0
    tl_mask = t2 > 0

    if not te_mask.any() or not tl_mask.any():
        return torch.zeros(line_points.shape[0], device=line_points.device, dtype=line_points.dtype)

    # d = n_i · p_i 
    d = torch.sum(plane_normals * plane_points, dim=1)  # (m,)

    # Compute all line-plane intersections at once
    lp_dot_n = torch.matmul(line_points, plane_normals.T)  # (n,m)
    numerator = d.unsqueeze(0) - lp_dot_n  # (n,m)
    t_i = numerator / t2.unsqueeze(0)

    # entry = max over negative denom planes, exit = min over positive denom planes
    t_e = torch.max(t_i[:, te_mask], dim=1).values
    t_l = torch.min(t_i[:, tl_mask], dim=1).values

    # Return clipped lengths, using zero for non-intersecting lines
    return torch.where(t_l > t_e + 1e-12, t_l - t_e, 0.0)


def alpha_to_quarternion(alpha_1, alpha_2, alpha_3):
    """Generate a unit quarternion by providing spherical angle coordinates on the S3 ball.

    Args:
        alpha_1,alpha_2,alpha_3 (:obj:`numpy array` or :obj:`float`): Radians. ``shape=(N,4)``

    Returns:
        (:obj:`numpy array`) Rotation as unit quarternion ``shape=(N,4)``

    """
    sin_alpha_1, sin_alpha_2 = np.sin(alpha_1), np.sin(alpha_2)
    return np.array(
        [
            np.cos(alpha_1),
            sin_alpha_1 * sin_alpha_2 * np.cos(alpha_3),
            sin_alpha_1 * sin_alpha_2 * np.sin(alpha_3),
            sin_alpha_1 * np.cos(alpha_2),
        ]
    ).T


def lab_strain_to_B_matrix(
    strain_tensor: torch.Tensor, crystal_orientation: torch.Tensor, B0: torch.Tensor
) -> torch.Tensor:
    """Take n strain tensors in lab coordinates and produce the lattice matrix (B matrix).

    Args:
        strain_tensor (:obj:`torch.Tensor`): Symmetric strain tensor in lab
            coordinates. ``shape=(n,3,3)``
        crystal_orientation (:obj:`torch.Tensor`): Unitary crystal orientation matrix.
            ``crystal_orientation`` maps from crystal to lab coordinates. ``shape=(n,3,3)``
        B0 matrix (:obj:`torch.Tensor`): Matrix containing the reciprocal underformed lattice parameters.``shape=(3,3)``

    Returns:
        (:obj:`torch.Tensor`) B matrix mapping from hkl Miller indices to realspace crystal
        coordinates. ``shape=(n,3,3)``

    """
    device = strain_tensor.device
    strain_tensor = ensure_torch(strain_tensor)
    crystal_orientation = ensure_torch(crystal_orientation)
    B0 = ensure_torch(B0)

    if strain_tensor.ndim == 2:
        strain_tensor = strain_tensor.unsqueeze(0)
    if crystal_orientation.ndim == 2:
        crystal_orientation = crystal_orientation.unsqueeze(0)

    crystal_strain = torch.matmul(
        crystal_orientation.transpose(1, 2),
        torch.matmul(strain_tensor, crystal_orientation),
    )
    B = _epsilon_to_b(crystal_strain, B0)
    return B.squeeze()


def _get_circumscribed_sphere_centroid(subset_of_points):
    """Compute circumscribed_sphere_centroid by solving linear systems of equations
    enforcing the centorid to be a linear combination of the subset_of_points space.

    The central idea is to substitute the squared radius in the nonlinear quadratic
    sphere equations and proceed to find a linear system with only the sphere centroid
    as the unknown.

    Args:
        subset_of_points (:obj:`numpy array`): Points to circumscribe with a sphere ``shape=(n,3)``

    Returns:
        (:obj:`numpy array`) with ``centroid`` of``shape=(3,)``

    """
    A = 2 * (subset_of_points[0] - subset_of_points[1:])
    pp = np.sum(subset_of_points * subset_of_points, axis=1)
    b = pp[0] - pp[1:]
    B = (subset_of_points[0] - subset_of_points[1:]).T
    x = np.linalg.solve(A.dot(B), b - A.dot(subset_of_points[0]))
    return subset_of_points[0] + B.dot(x)


def _set_xfab_logging(disabled):
    """Disable/Enable all loging of xfab; it is very verbose!"""
    xfab_modules = [
        "tools",
        "structure",
        "atomlib",
        "detector",
        "checks",
        "sg",
        "sglib",
        "symmetry",
    ]
    for sub_module in xfab_modules:
        logging.getLogger("xfab." + sub_module).disabled = disabled


class _verbose_manager(object):
    """Manage global verbose options in with statements; to turn of
    external package loggings easily inside xrd_simulator.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            _set_xfab_logging(disabled=False)

    def __exit__(self, type, value, traceback):
        if self.verbose:
            _set_xfab_logging(disabled=True)


def _strain_as_tensor(strain_vector):
    e11, e12, e13, e22, e23, e33 = strain_vector
    return np.asarray([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]], np.float64)


def _strain_as_vector(strain_tensor):
    return (
        list(strain_tensor[0, :]) + list(strain_tensor[1, 1:]) + [strain_tensor[2, 2]]
    )


def _b_to_epsilon(B_matrix, B0):
    """Handle large deformations as opposed to current xfab.tools.b_to_epsilon"""
    B = np.asarray(B_matrix, np.float64)
    F = np.dot(B0, np.linalg.inv(B))
    strain_tensor = 0.5 * (F.T.dot(F) - np.eye(3))  # large deformations
    return _strain_as_vector(strain_tensor)


def _epsilon_to_b(crystal_strain, B0):
    """Handle large deformations as opposed to current xfab.tools.epsilon_to_b"""
    device = crystal_strain.device
    crystal_strain = ensure_torch(crystal_strain)
    B0 = ensure_torch(B0)

    C = 2 * crystal_strain + torch.eye(3)

    eigen_vals = torch.linalg.eigvalsh(C)
    if torch.any(eigen_vals < 0):
        raise ValueError(
            "Unfeasible strain tensor with value: "
            + str(_strain_as_vector(crystal_strain))
            + ", will invert the unit cell with negative deformation gradient tensor determinant"
        )
    if C.ndim == 3:
        F = torch.transpose(torch.linalg.cholesky(C), 2, 1)
    else:
        F = torch.transpose(torch.linalg.cholesky(C), 1, 0)

    B = torch.matmul(torch.linalg.inv(F), B0)
    return B


def _get_misorientations(orientations):
    """
    Compute the minimal angles necessary to rotate a series of SO3 elements back into their mean orientation.

    Args:
        orientations (:obj: `numpy.array` or :obj: `torch.Tensor`): Orientation matrices, shape=(N,3,3)

    Returns:
        :obj: `numpy.array`: misorientations in units of radians, shape=(N,)
    """
    # Convert to numpy for scipy.spatial.transform.Rotation operations
    if torch.is_tensor(orientations):
        orientations_np = orientations.cpu().detach().numpy()
    else:
        orientations_np = orientations
        
    mean_orientation = Rotation.mean(Rotation.from_matrix(orientations_np)).as_matrix()
    misorientations = np.zeros((orientations_np.shape[0],))
    
    for i, U in enumerate(orientations_np):  # Use numpy array for iteration
        difference_rotation = Rotation.from_matrix(np.dot(U, mean_orientation.T))
        misorientations[i] = Rotation.magnitude(difference_rotation)
    
    return misorientations


def _compute_sides(points):
    """
    Computes the lengths of the sides of multiple tetrahedrons.

    Args:
        points (:obj: `numpy.array`): An array of shape (n, 4, 3), where `n` is the number of tetrahedrons.
                                Each tetrahedron is defined by 4 vertices in 3D space.

    Returns:
        :obj: `numpy.array`: An array of shape (n, 6) containing the lengths of the sides of the tetrahedrons.
                       Each row corresponds to a tetrahedron and contains the lengths of its 6 sides.
    """
    # Reshape the points array to have shape (n, 1, 4, 3)
    reshaped_points = points[:, np.newaxis, :, :]

    # Compute the differences between each pair of points
    differences = reshaped_points - reshaped_points.permute(0, 2, 1, 3)

    # Compute the squared distances along the last axis
    squared_distances = torch.sum(differences**2, dim=-1)

    # Compute the distances by taking the square root of the squared distances
    dist_mat = torch.sqrt(squared_distances)

    # Extract the 1-to-1 values from the distance matrix
    distances = torch.cat(
        (dist_mat[:, 0, 1:], dist_mat[:, 1, 2:], dist_mat[:, 2, 3].unsqueeze(1)),
        dim=1,
    )

    return distances


def _circumsphere_of_segments(segments):
    """
    Computes the circumcenters and circumradii of multiple line segments.

    Args:
        segments (:obj: `torch.Tensor`): A tensor of shape (n, 2, 3), where `n` is the number of line segments.
                                   Each line segment is defined by 2 vertices in 3D space.

    Returns:
        tuple(:obj: `torch.Tensor`, :obj: `torch.Tensor`): A tuple containing:
             - centers (:obj: `torch.Tensor`): A tensor of shape (n, 3) containing the circumcenters of the line segments.
             - radii (:obj: `torch.Tensor`): A tensor of shape (n,) containing the circumradii of the line segments.
    """
    centers = torch.mean(segments, dim=1)
    radii = torch.linalg.norm(centers - segments[:, 0, :], dim=1)
    return centers, radii * 1.0001  # because loss of floating point precision


def _circumsphere_of_triangles(triangles):
    """
    Computes the circumcenters and circumradii of multiple triangles.

    Args:
        triangles (:obj: `torch.Tensor`): A tensor of shape (n, 3, 3), where `n` is the number of triangles.
                                   Each triangle is defined by 3 vertices in 3D space.

    Returns:
        tuple(:obj: `torch.Tensor`, :obj: `torch.Tensor`): A tuple containing:
             - centers (:obj: `torch.Tensor`): A tensor of shape (n, 3) containing the circumcenters of the triangles.
             - radii (:obj: `torch.Tensor`): A tensor of shape (n,) containing the circumradii of the triangles.
    """
    ab = triangles[:, 1, :] - triangles[:, 0, :]
    ac = triangles[:, 2, :] - triangles[:, 0, :]

    abXac = torch.cross(ab, ac, dim=1)
    acXab = torch.cross(ac, ab, dim=1)

    a_to_centre = (
        torch.cross(abXac, ab, dim=1) * ((torch.linalg.norm(ac, dim=1) ** 2).unsqueeze(1))
        + torch.cross(acXab, ac, dim=1) * ((torch.linalg.norm(ab, dim=1) ** 2).unsqueeze(1))
    ) / (2 * (torch.linalg.norm(abXac, dim=1) ** 2).unsqueeze(1))

    centers = triangles[:, 0, :] + a_to_centre
    radii = torch.linalg.norm(a_to_centre, dim=1)

    return centers, radii * 1.0001  # because loss of floating point precision


def _circumsphere_of_tetrahedrons(tetrahedra):
    """
    Computes the circumcenters and circumradii of multiple tetrahedrons.

    Args:
        tetrahedra (:obj: `torch.Tensor`): A tensor of shape (n, 4, 3), where `n` is the number of tetrahedrons.
                                    Each tetrahedron is defined by 4 vertices in 3D space.

    Returns:
        tuple(:obj: `torch.Tensor`, :obj: `torch.Tensor`): A tuple containing:
             - centers (:obj: `torch.Tensor`): A tensor of shape (n, 3) containing the circumcenters of the tetrahedrons.
             - radii (:obj: `torch.Tensor`): A tensor of shape (n,) containing the circumradii of the tetrahedrons.
    """

    v0 = tetrahedra[:, 0, :]
    v1 = tetrahedra[:, 1, :]
    v2 = tetrahedra[:, 2, :]
    v3 = tetrahedra[:, 3, :]

    A = torch.stack(
        [
            (v1 - v0).T,
            (v2 - v0).T,
            (v3 - v0).T,
        ],
        dim=0
    ).permute(2, 0, 1)
    
    B = (
        0.5
        * torch.stack(
            [
                torch.linalg.norm(v1, dim=1) ** 2 - torch.linalg.norm(v0, dim=1) ** 2,
                torch.linalg.norm(v2, dim=1) ** 2 - torch.linalg.norm(v0, dim=1) ** 2,
                torch.linalg.norm(v3, dim=1) ** 2 - torch.linalg.norm(v0, dim=1) ** 2,
            ],
            dim=0
        ).T
    )

    centers = torch.matmul(torch.linalg.inv(A), B.unsqueeze(2)).squeeze(2)
    radii = torch.linalg.norm(
        (tetrahedra.permute(2, 0, 1) - centers.T.unsqueeze(2))[:, :, 0],
        dim=0,
    )

    return centers, radii * 1.0001  # because loss of floating point precision



def ensure_torch(data: np.ndarray | torch.Tensor | list | tuple, dtype=None) -> torch.Tensor:
    """Convert input to torch tensor if it isn't already.
    
    The device is automatically determined from torch's default device (set by torch.set_default_device).

    Args:
        data: Input data to convert. Can be:
            - numpy array
            - torch tensor
            - list
            - tuple
        dtype: Optional dtype for the tensor. If None, uses torch.float64.

    Returns:
        torch.Tensor: The input data converted to a torch tensor with specified dtype (default: float64)

    Examples:
        >>> ensure_torch([1, 2, 3])
        tensor([1., 2., 3.], dtype=torch.float64)
        >>> ensure_torch(np.array([1, 2, 3]), dtype=torch.int64)
        tensor([1, 2, 3], dtype=torch.int64)
        >>> ensure_torch(ensure_torch([1, 2, 3]))
        tensor([1., 2., 3.], dtype=torch.float64)
    """
    if torch is None:
        raise ImportError(
            "torch is required for ensure_torch but failed to import. "
            "See earlier import warnings for details."
        )

    # Automatically determine device from torch's default device
    # This respects torch.set_default_device() set by cuda.py
    device = torch.tensor(0.0).device

    if dtype is None:
        dtype = torch.float64

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device, dtype=dtype)
    elif torch.is_tensor(data):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, device=device)


def ensure_numpy(data: np.ndarray | torch.Tensor | list | tuple) -> np.ndarray:
    """Convert input to numpy array if it isn't already.

    Args:
        data: Input data to convert. Can be:
            - numpy array
            - torch tensor
            - list
            - tuple

    Returns:
        np.ndarray: The input data converted to a numpy array with float64 dtype

    Examples:
        >>> ensure_numpy([1, 2, 3])
        array([1., 2., 3.])
        >>> ensure_numpy(np.array([1, 2, 3]))
        array([1., 2., 3.])
        >>> ensure_numpy(ensure_torch([1, 2, 3]))
        array([1., 2., 3.])
    """
    if torch.is_tensor(data):
        if data.is_cuda:
            data = data.cpu()
        return data.detach().numpy().astype(np.float64)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    elif isinstance(data, (list, tuple)):
        return np.array(data, dtype=np.float64)
    return np.array(data, dtype=np.float64)

def print_memory_report(
    device_index: int = 0,
    include_cpu: bool = True,
    include_cuda: bool = True,
    limit: int | None = 5,
) -> Dict[str, float]:
    """Print VRAM/CPU tensor usage and list top tensors with names when available."""
    if torch is None:
        print("torch not available; no tensors to report.")
        return {"device": "none", "available_gb": 0.0, "free_gb": 0.0}

    # Inline VRAM summary (CUDA only)
    cuda_selected = get_selected_device() == "cuda"
    report = return_device_memory()

    if cuda_selected and include_cuda:
        device_props = torch.cuda.get_device_properties(device_index)
        total_memory = device_props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024**3)
        free_memory = total_memory - reserved
        utilization = (allocated / total_memory) * 100 if total_memory else 0.0

        print("\n" + "=" * 60)
        print(f"GPU MEMORY STATUS (device {device_index})")
        print("=" * 60)
        print(f"Total VRAM:      {total_memory:>8.1f} GB")
        print(f"Allocated:       {allocated:>8.1f} GB")
        print(f"Reserved:        {reserved:>8.1f} GB")
        print(f"Free:            {free_memory:>8.1f} GB")
        print(f"Utilization:     {utilization:>7.1f}%")
        print("=" * 60 + "\n")
    else:
        print("CUDA not available or excluded; showing CPU tensors only.")

    def _tensor_name(t: torch.Tensor) -> str:
        # Best-effort to find a referring dict key; may return multiple hits.
        try:
            for ref in gc.get_referrers(t):
                if isinstance(ref, dict):
                    for k, v in ref.items():
                        if v is t and isinstance(k, str):
                            return k
        except Exception:
            pass
        return f"<tensor@{hex(id(t))}>"

    # Collect tensors
    tensors: List[Dict[str, object]] = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                dev_type = obj.device.type
                if (dev_type == "cpu" and not include_cpu) or (dev_type == "cuda" and not include_cuda):
                    continue
                nbytes = obj.nelement() * obj.element_size()
                tensors.append(
                    {
                        "name": _tensor_name(obj),
                        "device": str(obj.device),
                        "dtype": str(obj.dtype),
                        "shape": tuple(obj.shape),
                        "bytes": nbytes,
                        "requires_grad": bool(getattr(obj, "requires_grad", False)),
                    }
                )
        except Exception:
            continue

    if not tensors:
        print("No tensors tracked.")
        return report

    tensors.sort(key=lambda x: x["bytes"], reverse=True)
    if limit is not None and limit > 0:
        tensors = tensors[:limit]

    # CPU summary
    if include_cpu:
        cpu_bytes = sum(t["bytes"] for t in tensors if "cpu" in t["device"])
        print(f"CPU tensors: {cpu_bytes / (1024**3):.3f} GB total (top {len([t for t in tensors if 'cpu' in t['device']])} listed)")

    print("Top tensors by size:")
    for idx, entry in enumerate(tensors, start=1):
        mb = entry["bytes"] / (1024**2)
        shape_str = "x".join(str(dim) for dim in entry["shape"])
        print(
            f"{idx:>2}. name={entry['name']:<20} device={entry['device']:<8} dtype={entry['dtype']:<12} "
            f"shape={shape_str:<20} size={mb:>8.1f} MB "
            f"grad={entry['requires_grad']}"
        )
    print()

    return report


def return_device_memory() -> Dict[str, float]:
    """Return available and free memory for the current device (CUDA or CPU)."""
    if torch is None:
        return {"device": "none", "available_gb": 0.0, "free_gb": 0.0}

    # Detect active backend
    backend = get_selected_device()
    if backend != "cuda":
        # CPU path
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            return {
                "device": "cpu",
                "available_gb": float(vm.total / (1024**3)),
                "free_gb": float(vm.available / (1024**3)),
            }
        except Exception:
            return {"device": "cpu", "available_gb": 0.0, "free_gb": 0.0}

    # CUDA path: use currently selected device
    try:
        device_index = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device_index)
        total_memory = device_props.total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024**3)
        free = total_memory - reserved
        return {
            "device": f"cuda:{device_index}",
            "available_gb": float(total_memory),
            "free_gb": float(free),
        }
    except Exception:
        return {"device": f"cuda:{torch.cuda.current_device() if torch.cuda.is_available() else 'unknown'}", "available_gb": 0.0, "free_gb": 0.0}


def compute_tetrahedra_volumes(vertices: torch.Tensor) -> torch.Tensor:
    """Compute volumes for multiple tetrahedra.

    Args:
        vertices: Tensor of shape (N, 4, 3) where:
            N is number of tetrahedra
            4 is number of vertices per tetrahedron
            3 is xyz coordinates

    Returns:
        torch.Tensor: Volumes of each tetrahedron, shape (N,)

    Example:
        >>> verts = torch.rand(10, 4, 3)  # 10 random tetrahedra
        >>> volumes = compute_tetrahedra_volumes(verts)
        >>> print(volumes.shape)
        torch.Size([10])
    """
    # Create vectors from first vertex to others (N, 3, 3)
    v1 = vertices[:, 0]  # Reference vertex (N, 3)
    v21 = vertices[:, 1] - v1  # (N, 3)
    v31 = vertices[:, 2] - v1
    v41 = vertices[:, 3] - v1

    # Stack vectors as columns (N, 3, 3)
    mat = torch.stack([v21, v31, v41], dim=2)

    # Compute determinant and divide by 6
    volumes = torch.abs(torch.linalg.det(mat)) / 6.0

    return volumes
