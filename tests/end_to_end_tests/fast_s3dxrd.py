import numpy as np
from xrd_simulator import templates
import matplotlib.pyplot as plt
import cProfile
import pstats
import os

from xrd_simulator.polycrystal import Polycrystal


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
path_to_cif_file = os.path.normpath( os.path.join(os.path.dirname(__file__), '..', 'data', 'quartz.cif') )
def orientation_density_function(x, q): return 1. / (np.pi**2)  # uniform ODF.

number_of_crystals = 50
sample_bounding_cylinder_height = 256 * 180 / 128.
sample_bounding_cylinder_radius = 256 * 180 / 128.
maximum_sampling_bin_seperation = np.radians(5.0)
# Linear strain gradient along rotation axis.

path = os.path.join(
    os.path.join(
        os.path.dirname(__file__),
        'saves'),
    'fast_polycrystal_from_odf.pc')

if 1:
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
    frames_to_render=0,
    lorentz=False,
    polarization=False,
    structure_factor=False,
    method="centroid",
    verbose=True)

def gaussian_kernel(side_length, sigma):
    ax = np.linspace(-(side_length - 1) / 2., (side_length - 1) / 2., side_length)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

from scipy.signal import convolve
diffraction_pattern[diffraction_pattern==0]=1
#kernel = gaussian_kernel(15, 0.5)
#diffraction_pattern = convolve(diffraction_pattern, kernel, mode='same', method='auto')
#plt.imshow(diffraction_pattern, cmap='jet')
plt.imshow(np.log(diffraction_pattern), cmap='jet')

plt.show()
