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


number_of_crystals = 50
# Larger sample dimensions for bigger crystals on detector
sample_bounding_cylinder_height = 256 * 180 / 16.
sample_bounding_cylinder_radius = 256 * 180 / 16.
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
                                             path_to_cif_file=None,
                                             maximum_sampling_bin_seperation=maximum_sampling_bin_seperation,
                                             strain_tensor=strain_tensor)
artifacts_dir = os.path.join(os.path.dirname(__file__), 'test_artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
path = os.path.join(artifacts_dir, 'polycrystal_from_odf.pc')
polycrystal.save(path, save_mesh_as_xdmf=True)
polycrystal = Polycrystal.load(path)

# Full field diffraction.
peaks_dict = polycrystal.diffract(
    beam,
    motion,
    min_bragg_angle=0,
    max_bragg_angle=None,
    detector=detector,
    verbose=True)
diffraction_pattern, peaks_dict = detector.render(
    peaks_dict,
    frames_to_render=1,
    method="macro")

pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(15)
print("")

# Convert to numpy for plotting
if hasattr(diffraction_pattern, 'cpu'):
    diffraction_pattern_np = diffraction_pattern[0].cpu().numpy()
else:
    diffraction_pattern_np = np.array(diffraction_pattern[0])

# Use log scale for better dynamic range visualization
diffraction_pattern_np[diffraction_pattern_np <= 0] = 1
plt.imshow(np.log(diffraction_pattern_np), cmap='jet')
plt.colorbar(label='log(intensity)')
plt.title('S3DXRD Diffraction Pattern')
plt.show()
