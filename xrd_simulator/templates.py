"""Templates module for fast creation of sample types and diffraction geometries.

The ``templates`` module allows for fast creation of a few select sample types
and diffraction geometries without having to worry about any of the "under the
hood" scripting.
"""

import numpy as np
import meshpy.tet as tet
import meshio
from scipy.spatial.transform import Rotation
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator import utils

PARAMETER_KEYS = [
    "detector_distance",
    "number_of_detector_pixels_z",
    "number_of_detector_pixels_y",
    "detector_center_pixel_z",
    "detector_center_pixel_y",
    "pixel_side_length_z",
    "pixel_side_length_y",
    "wavelength",
    "beam_side_length_z",
    "beam_side_length_y",
    "rotation_step",
    "rotation_axis",
]


def s3dxrd(parameters):
    """Construct a scanning-three-dimensional X-ray diffraction experiment.

    This is a helper/utility function for quickly creating an experiment. For
    full control over the diffraction geometry consider custom creation of the
    primitive quantities: :obj:`xrd_simulator.beam.Beam`, and
    :obj:`xrd_simulator.detector.Detector` separately.

    Parameters
    ----------
    parameters : dict
        Dictionary with the following fields:

        - ``"detector_distance"``: Distance from sample origin to detector
          centre in units of microns.
        - ``"number_of_detector_pixels_z"``: Number of detector pixels along
          z-axis.
        - ``"number_of_detector_pixels_y"``: Number of detector pixels along
          y-axis.
        - ``"detector_center_pixel_z"``: Intersection pixel coordinate between
          beam centroid line and detector along z-axis.
        - ``"detector_center_pixel_y"``: Intersection pixel coordinate between
          beam centroid line and detector along y-axis.
        - ``"pixel_side_length_z"``: Detector pixel side length in units of
          microns along z-axis.
        - ``"pixel_side_length_y"``: Detector pixel side length in units of
          microns along y-axis.
        - ``"wavelength"``: Wavelength in units of Angstrom.
        - ``"beam_side_length_z"``: Beam side length in units of microns.
        - ``"beam_side_length_y"``: Beam side length in units of microns.
        - ``"rotation_step"``: Angular frame integration step in units of
          radians.
        - ``"rotation_axis"``: Axis around which to positively rotate the
          sample by ``rotation_step`` radians.

    Returns
    -------
    tuple
        Objects defining an experiment:
        (:obj:`xrd_simulator.beam.Beam`,
        :obj:`xrd_simulator.detector.Detector`,
        :obj:`xrd_simulator.motion.RigidBodyMotion`).

    Examples
    --------
    .. literalinclude:: examples/example_s3dxrd.py
    """
    for key in PARAMETER_KEYS:
        if key not in list(parameters):
            raise ValueError(
                "No keyword " + key + " found in the input parameters dictionary"
            )

    detector = _get_detector_from_params(parameters)
    beam = _get_beam_from_params(parameters)
    motion = _get_motion_from_params(parameters)

    return beam, detector, motion


def polycrystal_from_odf(
    orientation_density_function,
    number_of_crystals,
    sample_bounding_cylinder_height,
    sample_bounding_cylinder_radius,
    unit_cell,
    sgname,
    path_to_cif_file=None,
    maximum_sampling_bin_seperation=np.radians(5.0),
    strain_tensor=lambda x: np.zeros((3, 3)),
):
    """Fill a cylinder with crystals from a given orientation density function.

    The ``orientation_density_function`` is sampled by discretizing orientation
    space over the unit quaternions. Each bin is assigned its appropriate
    probability, assuming the ``orientation_density_function`` is approximately
    constant over a single bin. Each sampled orientation is constructed by first
    drawing a random bin and next drawing uniformly from within that bin, again
    assuming that ``orientation_density_function`` is approximately constant
    over a bin.

    Parameters
    ----------
    orientation_density_function : callable
        Function ``orientation_density_function(x, q) -> float`` where input
        variable ``x`` is a numpy array of shape ``(3,)`` representing a
        spatial coordinate in the cylinder ``(x, y, z)`` and ``q`` is a numpy
        array of shape ``(4,)`` representing an orientation in SO3 by a unit
        quaternion. The format of the quaternion is "scalar last" (same as in
        ``scipy.spatial.transform.Rotation``).
    number_of_crystals : int
        Approximate number of crystal elements to compose the cylinder volume.
    sample_bounding_cylinder_height : float
        Height of sample cylinder in units of microns.
    sample_bounding_cylinder_radius : float
        Radius of sample cylinder in units of microns.
    unit_cell : list of float
        Crystal unit cell representation of the form
        ``[a, b, c, alpha, beta, gamma]``, where alpha, beta and gamma are in
        units of degrees while a, b and c are in units of Angstrom.
    sgname : str
        Name of space group, e.g. ``'P3221'`` for quartz, SiO2, for instance.
    path_to_cif_file : str, optional
        Path to CIF file. Default is ``None``, in which case no structure
        factors are computed.
    maximum_sampling_bin_seperation : float, optional
        Discretization steplength of orientation space using spherical
        coordinates over the unit quaternions in units of radians. A smaller
        steplength gives more accurate sampling of the input
        ``orientation_density_function`` but is computationally slower.
        Default is ``np.radians(5.0)``.
    strain_tensor : callable, optional
        Strain tensor field over sample cylinder.
        ``strain_tensor(x) -> numpy.ndarray`` of shape ``(3, 3)`` where input
        variable ``x`` is a numpy array of shape ``(3,)`` representing a
        spatial coordinate in the cylinder ``(x, y, z)``.
        Default returns zero strain.

    Returns
    -------
    Polycrystal
        A polycrystal sample filling the specified cylinder.

    Examples
    --------
    .. literalinclude:: examples/example_polycrystal_from_odf.py

    More complicated ODFs are also possible to use. Why not use a
    von-Mises-Fisher distribution for instance:

    .. literalinclude:: examples/example_polycrystal_from_von_mises_fisher_odf.py
    """
    # Sample topology
    volume_per_crystal = (
        np.pi
        * (sample_bounding_cylinder_radius**2)
        * sample_bounding_cylinder_height
        / number_of_crystals
    )
    max_cell_circumradius = (3 * volume_per_crystal / (np.pi * 4.0)) ** (1 / 3.0)

    # Fudge factor gives approximately number_of_crystals elements in the mesh
    # (calibrated for meshpy.tet)
    max_cell_circumradius = 4.2 * max_cell_circumradius

    dz = sample_bounding_cylinder_height / 2.0
    R = float(sample_bounding_cylinder_radius)
    
    # Generate points for the cylinder
    n = max(8, int(2 * np.pi * R / max_cell_circumradius))  # At least 8 points
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Bottom and top circles
    bottom_points = [(R*np.cos(angle), R*np.sin(angle), -dz) 
                    for angle in angles]
    top_points = [(R*np.cos(angle), R*np.sin(angle), dz) 
                  for angle in angles]
    
    # Add center points for top and bottom
    # Point indices: 0 to n-1 = bottom circle, n to 2n-1 = top circle, 
    # 2n = bottom center, 2n+1 = top center
    points = bottom_points + top_points + [(0, 0, -dz), (0, 0, dz)]
    
    # Create mesh info
    mesh_info = tet.MeshInfo()
    mesh_info.set_points(points)
    
    # Define triangular facets (not edges!)
    facets = []
    
    # Bottom circle triangles (from center to edge)
    bottom_center_idx = 2 * n
    for i in range(n):
        next_i = (i + 1) % n
        facets.append([bottom_center_idx, i, next_i])
    
    # Top circle triangles (from center to edge)
    top_center_idx = 2 * n + 1
    for i in range(n):
        next_i = (i + 1) % n
        # Reverse order for outward-facing normal
        facets.append([top_center_idx, n + next_i, n + i])
    
    # Side rectangles split into two triangles each
    for i in range(n):
        next_i = (i + 1) % n
        # Bottom triangle of rectangle
        facets.append([i, next_i, n + i])
        # Top triangle of rectangle
        facets.append([next_i, n + next_i, n + i])
    
    mesh_info.set_facets(facets)
    
    # Generate mesh
    mesh = tet.build(mesh_info, max_volume=(max_cell_circumradius**3)/6.0)
    
    # Convert to meshio format
    vertices = np.array(mesh.points)
    elements = np.array(mesh.elements)
    mesh_obj = meshio.Mesh(vertices, [("tetra", elements)])
    
    # Build TetraMesh
    mesh = TetraMesh._build_tetramesh(mesh_obj)

    # Sample is uniformly single phase
    phases = [Phase(unit_cell, sgname, path_to_cif_file)]
    element_phase_map = np.zeros((mesh.number_of_elements,)).astype(int)

    # Sample spatial texture
    orientation = _sample_ODF(
        orientation_density_function, maximum_sampling_bin_seperation, mesh.ecentroids
    )

    # Sample spatial strain
    strain_lab = np.zeros((mesh.number_of_elements, 3, 3))
    for ei in range(mesh.number_of_elements):
        # strain in lab/sample-coordinates
        strain_lab[ei] = strain_tensor(mesh.ecentroids[ei])

    return Polycrystal(
        mesh,
        orientation,
        strain=strain_lab,
        phases=phases,
        element_phase_map=element_phase_map,
    )


def get_uniform_powder_sample(
    sample_bounding_radius,
    number_of_grains,
    unit_cell,
    sgname,
    strain_tensor=np.zeros((3, 3)),
    path_to_cif_file=None,
):
    """Generate a polycrystal with grains overlayed at origin, uniform orientations.

    Parameters
    ----------
    sample_bounding_radius : float
        Bounding radius of sample. All tetrahedral crystal elements will be
        overlayed within a sphere of ``sample_bounding_radius`` radius.
    number_of_grains : int
        Number of grains composing the polycrystal sample.
    unit_cell : list of float
        Crystal unit cell representation of the form
        ``[a, b, c, alpha, beta, gamma]``, where alpha, beta and gamma are in
        units of degrees while a, b and c are in units of Angstrom.
    sgname : str
        Name of space group, e.g. ``'P3221'`` for quartz, SiO2, for instance.
    strain_tensor : numpy.ndarray, optional
        Strain tensor to apply to all tetrahedral crystal elements contained
        within the sample, shape ``(3, 3)``. Default is zero strain.
    path_to_cif_file : str, optional
        Path to CIF file. Default is ``None``, in which case no structure
        factors are computed.

    Returns
    -------
    Polycrystal
        A polycrystal sample with ``number_of_grains`` grains.

    Examples
    --------
    .. literalinclude:: examples/example_get_uniform_powder_sample.py
    """
    coord, enod, node_number = [], [], 0
    r = sample_bounding_radius
    for _ in range(number_of_grains):
        coord.append([r / np.sqrt(3.0), r / np.sqrt(3.0), -r / np.sqrt(3.0)])
        coord.append([r / np.sqrt(3.0), -r / np.sqrt(3.0), -r / np.sqrt(3.0)])
        coord.append([-r / np.sqrt(2.0), 0, -r / np.sqrt(2.0)])
        coord.append([0, 0, r])
        enod.append(list(range(node_number, node_number + 4)))
        node_number += 3
    coord, enod = np.array(coord), np.array(enod)
    mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)
    orientation = Rotation.random(mesh.number_of_elements).as_matrix()
    element_phase_map = np.zeros((mesh.number_of_elements,)).astype(int)
    phases = [Phase(unit_cell, sgname, path_to_cif_file)]
    return Polycrystal(
        mesh,
        orientation,
        strain=strain_tensor,
        phases=phases,
        element_phase_map=element_phase_map,
    )


def _get_motion_from_params(parameters):
    """Produce a ``xrd_simulator.motion.RigidBodyMotion`` from the s3dxrd params dictionary."""
    translation = np.array([0.0, 0.0, 0.0])
    return RigidBodyMotion(
        parameters["rotation_axis"], parameters["rotation_step"], translation
    )


def _get_beam_from_params(parameters):
    """Produce a ``xrd_simulator.beam.Beam`` from the s3dxrd params dictionary."""
    dz = parameters["beam_side_length_z"] / 2.0
    dy = parameters["beam_side_length_y"] / 2.0
    beam_vertices = np.array(
        [
            [-parameters["detector_distance"], -dy, -dz],
            [-parameters["detector_distance"], dy, -dz],
            [-parameters["detector_distance"], -dy, dz],
            [-parameters["detector_distance"], dy, dz],
            [parameters["detector_distance"], -dy, -dz],
            [parameters["detector_distance"], dy, -dz],
            [parameters["detector_distance"], -dy, dz],
            [parameters["detector_distance"], dy, dz],
        ]
    )

    beam_direction = np.array([1.0, 0.0, 0.0])
    polarization_vector = np.array([0.0, 1.0, 0.0])

    return Beam(
        beam_vertices, beam_direction, parameters["wavelength"], polarization_vector
    )


def _get_detector_from_params(parameters):
    """Produce a ``xrd_simulator.detector.Detector`` from the s3dxrd params dictionary."""

    p_z, p_y = parameters["pixel_side_length_z"], parameters["pixel_side_length_y"]
    det_dist = parameters["detector_distance"]
    det_pix_y = parameters["number_of_detector_pixels_y"]
    det_pix_z = parameters["number_of_detector_pixels_z"]
    det_cz, det_cy = (
        parameters["detector_center_pixel_z"],
        parameters["detector_center_pixel_y"],
    )

    d_0 = np.array([det_dist, -det_cy * p_y, -det_cz * p_z])
    d_1 = np.array([det_dist, (det_pix_y - det_cy) * p_y, -det_cz * p_z])
    d_2 = np.array([det_dist, -det_cy * p_y, (det_pix_z - det_cz) * p_z])

    return Detector(
        det_corner_0=d_0,
        det_corner_1=d_1,
        det_corner_2=d_2,
        pixel_size=(p_z, p_y))


def _sample_ODF(ODF, maximum_sampling_bin_seperation, coordinates):
    """Draw orientation matrices form an ODF at spatial locations ``coordinates``."""

    dalpha = maximum_sampling_bin_seperation / 2.0  # TODO: verify this analytically.
    dalpha = np.pi / 2.0 / int(np.pi / (dalpha * 2.0))
    alpha_1 = np.arange(
        0 + dalpha / 2.0 + 1e-8, np.pi / 2.0 - dalpha / 2.0 - 1e-8 + dalpha, dalpha
    )
    alpha_2 = np.arange(
        0 + dalpha / 2.0 + 1e-8, np.pi - dalpha / 2.0 - 1e-8 + dalpha, dalpha
    )
    alpha_3 = np.arange(
        0 + dalpha / 2.0 + 1e-8, 2 * np.pi - dalpha / 2.0 - 1e-8 + dalpha, dalpha
    )

    A1, A2, A3 = np.meshgrid(alpha_1, alpha_2, alpha_3, indexing="ij")
    A1, A2, A3 = A1.flatten(), A2.flatten(), A3.flatten()

    q = utils._alpha_to_quarternion(A1, A2, A3)

    # Approximate volume ber bin:
    # volume_element = (np.sin(A1)**2) * np.sin(A2) * (dalpha**3)

    # Actual volume per bin requires integration as below:
    da = dalpha / 2.0
    a = (1 / 2.0) * (A1 + da - np.sin(A1 + da) * np.cos(A1 + da)) - (1 / 2.0) * (
        A1 - da - np.sin(A1 - da) * np.cos(A1 - da)
    )
    b = -np.cos(A2 + da) + np.cos(A2 - da)
    c = (A3 + da) - (A3 - da)
    volume_element = a * b * c
    assert np.abs(np.sum(volume_element) - (np.pi**2)) < 1e-5

    rotations = []
    for x in coordinates:
        probability = volume_element * ODF(x, q)
        assert (
            np.abs(np.sum(probability) - 1.0) < 0.05
        ), "Orientation density function must be be normalised."
        # Normalisation is not exact due to the discretization.
        probability = probability / np.sum(probability)
        indices = np.linspace(0, len(probability) - 1, len(probability)).astype(int)
        draw = np.random.choice(indices, size=1, replace=True, p=probability)
        a1 = A1[draw] + dalpha * (np.random.rand() - 0.5)
        a2 = A2[draw] + dalpha * (np.random.rand() - 0.5)
        a3 = A3[draw] + dalpha * (np.random.rand() - 0.5)
        q_pertubated = utils._alpha_to_quarternion(a1, a2, a3)
        rotations.append(Rotation.from_quat(q_pertubated).as_matrix().reshape(3, 3))

    return np.array(rotations)
