import numpy as np
from xrd_simulator import templates
import matplotlib.pyplot as plt
import cProfile
import pstats
import os

from xrd_simulator.polycrystal import Polycrystal

pr = cProfile.Profile()
pr.enable()

parameters = {
    "detector_distance": 191023.9164,
    "detector_center_pixel_z": 1024.938,
    "detector_center_pixel_y": 1020.4516,
    "pixel_side_length_z": 45.35585,
    "pixel_side_length_y": 45.058575,
    "number_of_detector_pixels_z": 2048,
    "number_of_detector_pixels_y": 2048,
    "wavelength": 0.285227,
    "beam_side_length_z": 102400.0,
    "beam_side_length_y": 102400.0,
    "rotation_step": np.radians(2.0),
    "rotation_axis": np.array([0., 0., 1.0])
}

beam, detector, motion = templates.s3dxrd(parameters)

unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
sgname = 'P3221'  # Quartz
path_to_cif_file = None
def orientation_density_function(x, q): return 1. / (np.pi**2)  # uniform ODF.


number_of_crystals = 10000
sample_bounding_cylinder_height = parameters['pixel_side_length_z']
sample_bounding_cylinder_radius = parameters['pixel_side_length_y']
maximum_sampling_bin_seperation = np.radians(10.0)

# Linear strain gradient along rotation axis.
def strain_tensor(x): return np.array(
    [[0, 0, 0], [0, 0, 0], [0, 0, 0.02 * x[2] / sample_bounding_cylinder_height]])

polycrystal = templates.polycrystal_from_odf(orientation_density_function,
                                             number_of_crystals,
                                             sample_bounding_cylinder_height,
                                             sample_bounding_cylinder_radius,
                                             unit_cell,
                                             sgname,
                                             path_to_cif_file,
                                             maximum_sampling_bin_seperation,
                                             strain_tensor)
path = os.path.join(
    os.path.join(
        os.path.dirname(__file__),
        'saves'),
    'polycrystal_from_odf.pc')
polycrystal.save(path, save_mesh_as_xdmf=True)
polycrystal = Polycrystal.load(path)

# Full-field diffraction.
polycrystal.diffract(
    beam,
    detector,
    motion,
    min_bragg_angle=0,
    max_bragg_angle=None,
    verbose=True,
    proximity=True,
    BB_intersection=True)

diffraction_pattern = detector.render(
    frames_to_render=0,
    lorentz=False,
    polarization=False,
    structure_factor=False,
    method="centroid_with_scintillator",
    verbose=True)

pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(15)
print("")

plt.imshow(diffraction_pattern, cmap='jet')
plt.show()
