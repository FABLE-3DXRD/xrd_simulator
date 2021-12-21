import numpy as np
from scipy.spatial.transform import Rotation

IMAGE_D11_PARAM_KEYS=[
"cell__a",
"cell__b",
"cell__c",
"cell_alpha",
"cell_beta",
"cell_gamma",
"cell_lattice_[P,A,B,C,I,F,R],
"chi",
"distance",
"fit_tol",
"fit_tolerance",
"min_bin_prob",
"no_bins",
"o11",
"o12",
"o21",
"o22",
"omegasign",
"t_x",
"t_y",
"t_z",
"tilt_x",
"tilt_y",
"tilt_z",
"wavelength",
"wedge",
"y_center",
"y_size",
"z_center",
"z_size"
]

def s3dxrd( detector_distance, detector_size_y, detector_size_z, pixel_size_z, pixel_size_y, beam_size ):
    """Construct a scaning-three-dimensional-xray diffraction experiment.

    This is a helper/utility function for quickly creating an experiment. For full controll
    over the diffraction geometry consider custom creation of the primitive quanteties: 
    (:obj:`xrd_simulator.beam.Beam`), (:obj:`xrd_simulator.polycrystal.Polycrystal`), 
    (:obj:`xrd_simulator.phase.Phase`), and (:obj:`xrd_simulator.detector.Detector`) seperately.

    Args:
        detector_distance (:obj:`float`): Sample origin to detector distance in units of microns.
        detector_size_z (:obj:`float`): Detector side length along rotation axis in units of microns.
        detector_size_y (:obj:`float`): Detector side length perpendicular to rotation axis in units of microns.
        pixel_size_z (:obj:`float`): Pixel side length along rotation axis in units of microns.
        pixel_size_y (:obj:`float`): Pixel side length perpendicular to rotation axis in units of microns.
        beam_size (:obj:`float`): Side length of beam (square cross-section) in units of microns.

    Returns:
        (:obj:`xrd_simulator`) objects defining an experiment:
        (:obj:`xrd_simulator.beam.Beam`),
        (:obj:`xrd_simulator.polycrystal.Polycrystal`), 
        (:obj:`xrd_simulator.phase.Phase`) and 
        (:obj:`xrd_simulator.detector.Detector`).

    """powder_3dxrd
    raise NotImplementedError()

def powder_3dxrd( parameter_file_path, ODF ):
    """Construct a simplified full field illumination three-dimensional-xray diffraction experiment with a powder sample.

    This is a helper/utility function for quickly creating an experiment. For full controll
    over the diffraction geometry consider custom creation of the primitive quanteties: 
    (:obj:`xrd_simulator.beam.Beam`), (:obj:`xrd_simulator.polycrystal.Polycrystal`), 
    (:obj:`xrd_simulator.phase.Phase`), and (:obj:`xrd_simulator.detector.Detector`) seperately.

    Args:
        parameter_file_path (:obj:`string`): Sample origin to detector distance in units of microns.
        ODF (:obj:`float`): 

    Returns:
        (:obj:`xrd_simulator`) objects defining an experiment:
        (:obj:`xrd_simulator.beam.Beam`),
        (:obj:`xrd_simulator.polycrystal.Polycrystal`), 
        (:obj:`xrd_simulator.phase.Phase`) and 
        (:obj:`xrd_simulator.detector.Detector`).

    """
    raise NotImplementedError()


def sample_orientation_density( orientation_density_function, 
                                number_of_crystals, 
                                sample_bounding_cylinder_height, 
                                sample_bounding_cylinder_radius  ):
    """Fill a cylinder with crystals from a given orientation density function. 
    """
    raise NotImplementedError()