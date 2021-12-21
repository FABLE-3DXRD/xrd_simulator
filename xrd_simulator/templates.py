import numpy as np
from scipy.spatial.transform import Rotation
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.mesh import TetraMesh

PARAMETER_KEYS = [
"detector_distance",
"number_of_detector_pixels_y",
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
        parameters (:obj:`dict`):

    Returns:
        (:obj:`xrd_simulator`) objects defining an experiment:
        (:obj:`xrd_simulator.beam.Beam`),
        (:obj:`xrd_simulator.detector.Detector`).

    """

    for key in PARAMETER_KEYS:
        if key not in list(parameters):
            raise ValueError("No keyword "+key+" found in the input parameters dictionary")

    detector = _get_detector_from_params( parameters )
    beam     = _get_detector_from_params( parameters )
    motion   = _get_motion_from_params( parameters )

    return beam, detector, motion

def _get_motion_from_params( parameters ):
    translation = np.array([0., 0., 0.])
    return RigidBodyMotion(parameters["rotation_axis"], parameters["rotation_step"],translation)

def _get_beam_from_params( parameters ):

    dz = parameters['beam_side_length_z']/2.
    dy = parameters['beam_side_length_y']/2. 
    beam_vertices = np.array([
        [-parameters['detector_distance'], -dy, -dz ],
        [-parameters['detector_distance'],  dy, -dz ],
        [-parameters['detector_distance'], -dy,  dz ],
        [ parameters['detector_distance'], -dy, -dz ],
        [ parameters['detector_distance'],  dy, -dz ],
        [ parameters['detector_distance'], -dy,  dz ],
        [ parameters['detector_distance'],  dy,  dz ]
    ])

    beam_direction      = np.array([1.0, 0.0, 0.0])
    polarization_vector = np.array([1.0, 0.0, 0.0])

    return Beam(beam_vertices, beam_direction, parameters['wavelength'], polarization_vector)

def _get_detector_from_params( parameters ):

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

def polycrystal_from_orientation_density( orientation_density_function,
                                          number_of_crystals,
                                          sample_bounding_cylinder_height,
                                          sample_bounding_cylinder_radius,                                          
                                          unit_cell,
                                          sgname  ):
    """Fill a cylinder with crystals from a given orientation density function. 
    """

    def level_set(x):
        return np.min( [x[0]**2 + x[1]**2 - sample_bounding_cylinder_radius**2, (sample_bounding_cylinder_height/2.) - np.abs(x[2])] )
    bounding_radius       = np.max([sample_bounding_cylinder_radius, (sample_bounding_cylinder_height/2.)])
    volume_per_crystal    = np.pi*(sample_bounding_cylinder_radius**2)*sample_bounding_cylinder_height / number_of_crystals
    max_cell_circumradius = 100*( 3 * volume_per_crystal / (np.pi*4.) )**(1/3.)
    print(max_cell_circumradius)

    import pygalmesh

    c = pygalmesh.Cylinder(-1.0, 1.0, 0.7, 0.4)
    cylinder = pygalmesh.generate_mesh(
        c, max_cell_circumradius=0.4, max_edge_size_at_feature_edges=0.4, verbose=False
    )
    mesh = TetraMesh._build_tetramesh(cylinder)
    mesh.save("/home/axel/Downloads/cylinder.xdmf")

    raise
    mesh = TetraMesh.generate_mesh_from_levelset( level_set, bounding_radius, max_cell_circumradius)

    eU = _sample_ODF( orientation_density_function, np.pi/90.0, mesh.ecentroids )
    phases = [Phase(unit_cell, sgname)]
    B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
    eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )
    ephase = np.zeros((mesh.number_of_elements,)).astype(int)
    polycrystal = Polycrystal(mesh, ephase, eU, eB, phases)

def _sample_ODF( ODF, dalpha, coordinates ):
    """
    """

    alpha_1 = np.arange(  0,   np.pi,   dalpha )
    alpha_2 = np.arange(  0,   np.pi,   dalpha )
    alpha_3 = np.arange(  0, 2*np.pi,   dalpha )

    A1,A2,A3 = np.meshgrid( alpha_1, alpha_2, alpha_3, indexing='ij' )
    A1,A2,A3 = A1.flatten(), A2.flatten(), A3.flatten()

    q              = _alpha_to_quarternion(A1, A2, A3)
    volume_element = (np.sin(A1)**2) * np.sin(A2) * (dalpha**3)

    rotations = []
    for x in coordinates:
        probability = jac * function( x, q ) * (dalpha**3)
        indices = np.linspace(0, len(probability)).astype(int)
        draw = np.random.choice(indices, size=1, replace=True, p=probability)
        rotations.append( Rotation.as_rotmat( q[draw,:] ) )

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
