import numpy as np
import pygalmesh
from scipy.spatial.transform import Rotation
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xfab import tools

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
"rotation_axis"
]

def s3dxrd( parameters ):
    """Construct a scaning-three-dimensional-xray diffraction experiment.

    This is a helper/utility function for quickly creating an experiment. For full controll
    over the diffraction geometry consider custom creation of the primitive quanteties: 
    (:obj:`xrd_simulator.beam.Beam`), and (:obj:`xrd_simulator.detector.Detector`) seperately.

    Args:
        parameters (:obj:`dict`): Dictionary with fields as \n
            ``"detector_distance"``           : (:obj:`float`) Distance form sample origin to detector centre in units of microns. \n
            ``"number_of_detector_pixels_z"`` : (:obj:`int`) Number of detector pixels along z-axis. \n
            ``"number_of_detector_pixels_y"`` : (:obj:`int`) Number of detector pixels along y-axis. \n
            ``"detector_center_pixel_z"``     : (:obj:`float`) Intersection pixel coordinate between beam centroid line and detector along z-axis. \n
            ``"detector_center_pixel_y"``     : (:obj:`float`) Intersection pixel coordinate between beam centroid line and detector along y-axis. \n
            ``"pixel_side_length_z"``         : (:obj:`float`) Detector pixel side length in units of microns along z-axis. \n
            ``"pixel_side_length_y"``         : (:obj:`float`) Detector pixel side length in units of microns along y-axis. \n
            ``"wavelength"``                  : (:obj:`float`) Wavelength in units of Angstrom. \n
            ``"beam_side_length_z"``          : (:obj:`float`) Beam side length in units of microns. \n
            ``"beam_side_length_y"``          : (:obj:`float`) Beam side length in units of microns. \n
            ``"rotation_step"``               : (:obj:`float`) Angular frame integration step in units of radians. \n
            ``"rotation_axis"``               : (:obj:`numpy array`)  Axis around which to positively rotate the sample by ``rotation_step`` radians. \n

    Returns:
        (:obj:`xrd_simulator`) objects defining an experiment:
        (:obj:`xrd_simulator.beam.Beam`), (:obj:`xrd_simulator.detector.Detector`), (:obj:`xrd_simulator.motion.RigidBodyMotion`).

    Examples:
        .. literalinclude:: examples/s3dxrd.py

    """

    for key in PARAMETER_KEYS:
        if key not in list(parameters):
            raise ValueError("No keyword "+key+" found in the input parameters dictionary")

    detector = _get_detector_from_params( parameters )
    beam     = _get_beam_from_params( parameters )
    motion   = _get_motion_from_params( parameters )

    return beam, detector, motion

def _get_motion_from_params( parameters ):
    """Produce a ``xrd_simulator.motion.RigidBodyMotion`` from the s3dxrd params dictionary.
    """
    translation = np.array([0., 0., 0.])
    return RigidBodyMotion(parameters["rotation_axis"], parameters["rotation_step"],translation)

def _get_beam_from_params( parameters ):
    """Produce a ``xrd_simulator.beam.Beam`` from the s3dxrd params dictionary.
    """
    dz = parameters['beam_side_length_z']/2.
    dy = parameters['beam_side_length_y']/2. 
    beam_vertices = np.array([
        [-parameters['detector_distance'], -dy, -dz ],
        [-parameters['detector_distance'],  dy, -dz ],
        [-parameters['detector_distance'], -dy,  dz ],
        [-parameters['detector_distance'],  dy,  dz ],
        [ parameters['detector_distance'], -dy, -dz ],
        [ parameters['detector_distance'],  dy, -dz ],
        [ parameters['detector_distance'], -dy,  dz ],
        [ parameters['detector_distance'],  dy,  dz ]
    ])

    beam_direction      = np.array([1.0, 0.0, 0.0])
    polarization_vector = np.array([0.0, 1.0, 0.0])

    return Beam(beam_vertices, beam_direction, parameters['wavelength'], polarization_vector)

def _get_detector_from_params( parameters ):
    """Produce a ``xrd_simulator.detector.Detector`` from the s3dxrd params dictionary.
    """
    d0 = np.array( [  parameters['detector_distance'], 
                     -parameters['detector_center_pixel_y']  * parameters['pixel_side_length_y'], 
                     -parameters['detector_center_pixel_z']  * parameters['pixel_side_length_z']
                    ] )

    d1 = np.array( [ parameters['detector_distance'],  
                     (parameters['number_of_detector_pixels_y'] - parameters['detector_center_pixel_y']) * parameters['pixel_side_length_y'], 
                    -parameters['detector_center_pixel_z']  * parameters['pixel_side_length_z']
                    ] ) 

    d2 = np.array( [ parameters['detector_distance'], 
                    -parameters['detector_center_pixel_y']  * parameters['pixel_side_length_y'],  
                     (parameters['number_of_detector_pixels_z'] - parameters['detector_center_pixel_z']) * parameters['pixel_side_length_z']
                     ] )

    return Detector( parameters['pixel_side_length_z'], parameters['pixel_side_length_y'], d0, d1, d2 )

def polycrystal_from_odf( orientation_density_function,
                          number_of_crystals,
                          sample_bounding_cylinder_height,
                          sample_bounding_cylinder_radius,                                          
                          unit_cell,
                          sgname,
                          maximum_sampling_bin_seperation=np.radians(5.0),
                          strain_tensor=lambda x: np.zeros((3,3))):
    """Fill a cylinder with crystals from a given orientation density function. 

    The ``orientation_density_function`` is sampled by discretizing orientation space over the unit
    quarternions. Each bin is assigned its aproiate probability, assuming the ``orientation_density_function``
    is approximately constant over a single bin. Each sampled orientation then corresponds is first drawing a
    random bin and next drawing unifromly from within that bin, again assuming that ``orientation_density_function``
    is approximately constant over a bin.

    Args:
        orientation_density_function (:obj:`callable`): orientation_density_function(x, q) -> :obj:`float` where input
            variable ``x`` is a :obj:`numpy array` of shape ``(3,)`` representing a spatial coordinate in the cylinder
            (x,y,z) and ``q`` is a :obj:`numpy array` of shape ``(4,)`` representing a orientaiton in so3 by a unit 
            quarternion. The format of the quarternion is "scalar last" (same as in scipy.spatial.transform.Rotation).
        number_of_crystals (:obj:`int`): Approximate number of crystal elements to compose the cylinder volume.
        sample_bounding_cylinder_height (:obj:`float`): Height of sample cylinder in units of microns.
        sample_bounding_cylinder_radius (:obj:`float`): Radius of sample cylinder in units of microns.
        unit_cell (:obj:`list` of :obj:`float`): Crystal unit cell representation of the form 
            [a,b,c,alpha,beta,gamma], where alpha,beta and gamma are in units of degrees while
            a,b and c are in units of anstrom.
        sgname (:obj:`string`): Name of space group , e.g 'P3221' for quartz, SiO2, for instance
        maximum_sampling_bin_seperation (:obj:`float`): Discretization steplenght of orientaiton space using spherical coordinates
            over the unit quarternions in units of radians. A smaller steplenght gives more accurate sampling of the input
            ``orientation_density_function`` but is computationally slower.
        strain_tensor (:obj:`callable`): Strain tensor field over sample cylinder. strain_tensor(x) -> :obj:`numpy array` of shape ``(3,3)``
            where input variable ``x`` is a :obj:`numpy array` of shape ``(3,)`` representing a spatial coordinate in the cylinder (x,y,z).

    Returns:
        (:obj:`xrd_simulator.polycrystal.Polycrystal`)

    Examples:
        .. literalinclude:: examples/polycrystal_from_odf.py

    """
    # Sample topology
    volume_per_crystal    = np.pi*(sample_bounding_cylinder_radius**2)*sample_bounding_cylinder_height / number_of_crystals
    max_cell_circumradius = ( 3 * volume_per_crystal / (np.pi*4.) )**(1/3.)

    # Fudge factor 2.6 gives approximately number_of_crystals elements in the mesh
    max_cell_circumradius = 2.65 * max_cell_circumradius 

    dz = sample_bounding_cylinder_height/2.
    R = float(sample_bounding_cylinder_radius)

    cylinder = pygalmesh.generate_mesh(
        pygalmesh.Cylinder(-dz, dz, R, max_cell_circumradius), 
        max_cell_circumradius=max_cell_circumradius, 
        max_edge_size_at_feature_edges=max_cell_circumradius, 
        verbose=False
    )
    mesh = TetraMesh._build_tetramesh( cylinder )

    # Sample is uniformly single phase
    phases = [Phase(unit_cell, sgname)]
    ephase = np.zeros((mesh.number_of_elements,)).astype(int)

    # Sample spatial texture
    eU = _sample_ODF( orientation_density_function, maximum_sampling_bin_seperation, mesh.ecentroids )

    # Sample spatial strain
    eB = np.zeros( (mesh.number_of_elements,3,3) )
    for ei in range(mesh.number_of_elements):
        strain_lab = strain_tensor( mesh.ecentroids[ei] )  # strain in lab-coordinates
        s = np.dot( eU[ei].T, np.dot(strain_lab, eU[ei]) ) # strain in crystal coordinate system
        strain = [ s[0,0], s[0,1], s[0,2], s[1,1], s[1,2], s[2,2] ]
        eB[ei] =  tools.epsilon_to_b( strain, unit_cell ) 

    return Polycrystal(mesh, ephase, eU, eB, phases)

def _sample_ODF( ODF, maximum_sampling_bin_seperation, coordinates ):
    """Draw orientation matrices form an ODF at spatial locations ``coordinates``.
    """

    dalpha  = maximum_sampling_bin_seperation / 2. #TODO: verify this analytically.
    dalpha = np.pi/2. / int( np.pi/(dalpha * 2.) )
    alpha_1 = np.arange(  0 + dalpha/2. + 1e-8,   np.pi/2. - dalpha/2. - 1e-8 + dalpha,    dalpha )
    alpha_2 = np.arange(  0 + dalpha/2. + 1e-8,   np.pi    - dalpha/2. - 1e-8 + dalpha,    dalpha )
    alpha_3 = np.arange(  0 + dalpha/2. + 1e-8,   2*np.pi  - dalpha/2. - 1e-8 + dalpha,    dalpha )

    A1,A2,A3 = np.meshgrid( alpha_1, alpha_2, alpha_3, indexing='ij' )
    A1,A2,A3 = A1.flatten(), A2.flatten(), A3.flatten()

    q = _alpha_to_quarternion(A1, A2, A3)

    # Approximate volume ber bin: 
    # volume_element = (np.sin(A1)**2) * np.sin(A2) * (dalpha**3)

    # Actual volume per bin requires integration as below:
    da = dalpha/2.
    a = ( (1/2.)*(A1+da - np.sin(A1+da)*np.cos(A1+da)) - (1/2.)*(A1-da - np.sin(A1-da)*np.cos(A1-da)) )
    b = ( -np.cos(A2+da) + np.cos(A2-da) )
    c = ( (A3+da) - (A3-da) )
    volume_element = a * b * c
    assert np.abs(np.sum(volume_element)-(np.pi**2))<1e-5

    rotations = []
    for x in coordinates:
        probability = volume_element * ODF( x, q )
        assert np.abs(np.sum(probability)-1.0)<0.05, "Orientation density function must be be normalised." 
        probability = probability / np.sum(probability) # Normalisation is not exact due to the discretization.
        indices = np.linspace( 0, len(probability)-1, len(probability) ).astype(int)
        draw = np.random.choice(indices, size=1, replace=True, p=probability)
        a1 = A1[draw] + dalpha * (np.random.rand()-0.5)
        a2 = A2[draw] + dalpha * (np.random.rand()-0.5)
        a3 = A3[draw] + dalpha * (np.random.rand()-0.5)
        q_pertubated = _alpha_to_quarternion(a1, a2, a3)
        rotations.append( Rotation.from_quat( q_pertubated ).as_matrix().reshape(3,3) )

    return np.array(rotations)

def _alpha_to_quarternion(alpha_1, alpha_2, alpha_3):
    """Generate a unit quarternion by providing spherical angle coordinates on the S3 ball.
    """
    sa1,sa2 = np.sin(alpha_1),np.sin(alpha_2)
    x = np.cos(alpha_1) 
    y = sa1*sa2*np.cos(alpha_3)
    z = sa1*sa2*np.sin(alpha_3)
    w = sa1*np.cos(alpha_2)
    return np.array([x,y,z,w]).T

def get_uniform_powder_sample( 
    sample_bounding_radius, 
    number_of_grains, 
    unit_cell,
    sgname,
    strain_tensor = np.zeros((3,3)) ):
    """Generate a polycyrystal with all grains overlayed at the origin and orientations drawn uniformly.

    Args:
        sample_bounding_radius (:obj:`float`): Bounding radius of sample. All tetrahedral crystal elements
            will be overlayed within a sphere of ``sample_bounding_radius`` radius.
        number_of_grains (:obj:`int`): Number of grains composing the polycrystal sample.
        unit_cell (:obj:`list` of :obj:`float`): Crystal unit cell representation of the form 
            [a,b,c,alpha,beta,gamma], where alpha,beta and gamma are in units of degrees while
            a,b and c are in units of anstrom.
        sgname (:obj:`string`):  Name of space group , e.g 'P3221' for quartz, SiO2, for instance
        strain_tensor (:obj:`numpy array`): Strain tensor to apply to all tetrahedral crystal elements
            contained within the sample. ``shape=(3,3)``.

    Returns:
        (:obj:`xrd_simulator.polycrystal`) A polycyrystal sample with ``number_of_grains`` grains.

    Examples:
        .. literalinclude:: examples/get_uniform_powder_sample.py

    """
    coord, enod, eU, node_number = [],[],[],0
    r = sample_bounding_radius
    for _ in range(number_of_grains):
        coord.append( [ r/np.sqrt(3.),   r/np.sqrt(3.),   -r/np.sqrt(3.)] ) 
        coord.append( [ r/np.sqrt(3.),  -r/np.sqrt(3.),   -r/np.sqrt(3.)] ) 
        coord.append( [-r/np.sqrt(2.),       0,           -r/np.sqrt(2.)] ) 
        coord.append( [       0,             0,                 r       ] )
        enod.append( list( range( node_number, node_number+4 ) ) )
        node_number+=3
    coord, enod = np.array(coord), np.array(enod)
    mesh = TetraMesh.generate_mesh_from_vertices(coord,enod)

    B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
    eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )
    eU = Rotation.random(mesh.number_of_elements).as_matrix()
    ephase = np.zeros((mesh.number_of_elements,)).astype(int)
    return Polycrystal(mesh, ephase, eU, eB, [Phase(unit_cell, sgname)])
