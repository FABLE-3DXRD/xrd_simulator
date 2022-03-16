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
    "detector_center_pixel_z": 256.2345,
    "detector_center_pixel_y": 255.1129,
    "pixel_side_length_z": 181.4234,
    "pixel_side_length_y": 180.2343,
    "number_of_detector_pixels_z": 512,
    "number_of_detector_pixels_y": 512,
    "wavelength": 0.285227,
    "beam_side_length_z": 512 * 200.,
    "beam_side_length_y": 512 * 200.,
    "rotation_step": np.radians(10.0),
    "rotation_axis": np.array([0., 0., 1.0])
}

beam, detector, motion = templates.s3dxrd(parameters)

unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
sgname = 'P3221'  # Quartz
def orientation_density_function(x, q): return 1. / (np.pi**2)  # uniform ODF.


number_of_crystals = 500
sample_bounding_cylinder_height = 256 * 180 / 128.
sample_bounding_cylinder_radius = 256 * 180 / 128.
maximum_sampling_bin_seperation = np.radians(10.0)
# Linear strain gradient along rotation axis.


def strain_tensor(x): return np.array(
    [[0, 0, 0], [0, 0, 0], [0, 0, 0.02 * x[2] / sample_bounding_cylinder_height]])

# Make the beam much smaller than the sample
# vertices = beam.vertices.copy()
# vertices[:,1:] = 0.180*vertices[:,1:]/np.max(vertices[:,1:])
# beam.set_beam_vertices(vertices)


polycrystal = templates.polycrystal_from_odf(orientation_density_function,
                                             number_of_crystals,
                                             sample_bounding_cylinder_height,
                                             sample_bounding_cylinder_radius,
                                             unit_cell,
                                             sgname,
                                             maximum_sampling_bin_seperation,
                                             strain_tensor)
path = os.path.join(
    os.path.join(
        os.path.dirname(__file__),
        'saves'),
    'polycrystal_from_odf')
polycrystal.save(path, save_mesh_as_xdmf=True)
polycrystal = Polycrystal.load(path)

# Full field diffraction.
polycrystal.diffract(
    beam,
    detector,
    motion,
    min_bragg_angle=0,
    max_bragg_angle=None,
    verbose=True)
diffraction_pattern = detector.render(
    frame_number=0,
    lorentz=False,
    polarization=False,
    structure_factor=False,
    method="centroid",
    verbose=True)

pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(15)
print("")

diffraction_pattern[diffraction_pattern > 0] = 1
plt.imshow(diffraction_pattern, cmap='gray')
plt.show()
