import matplotlib.pyplot as plt
import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xfab import tools
import os
import cProfile
import pstats

np.random.seed(23)

grainmeshfile = os.path.join(
    os.path.join(
        os.path.dirname(__file__),
        '../data'),
    'grain0056.xdmf')
mesh = TetraMesh.load(grainmeshfile)

sample_diameter = 1.0

print("")
print('nelm:', mesh.number_of_elements)
print("")

pixel_size = sample_diameter / 256.
detector_size = pixel_size * 1024
detector_distance = 10 * sample_diameter
det_corner_0 = np.array(
    [detector_distance, -detector_size / 2., -detector_size / 2.])
det_corner_1 = np.array(
    [detector_distance, detector_size / 2., -detector_size / 2.])
det_corner_2 = np.array(
    [detector_distance, -detector_size / 2., detector_size / 2.])

detector = Detector(
    pixel_size,
    pixel_size,
    det_corner_0,
    det_corner_1,
    det_corner_2)

# data = os.path.join( os.path.join(os.path.dirname(__file__), 'data' ), 'Fe_mp-150_conventional_standard.cif' )
unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
sgname = 'Fm-3m'  # Iron
phases = [Phase(unit_cell, sgname)]

grain_avg_rot = np.max([np.radians(1.0), np.random.rand() * 2 * np.pi])
euler_angles = grain_avg_rot + \
    np.random.normal(loc=0.0, scale=np.radians(0.01), size=(mesh.number_of_elements, 3))
orientation = np.array([tools.euler_to_u(ea[0], ea[1], ea[2])
                       for ea in euler_angles])
element_phase_map = np.zeros((mesh.number_of_elements,)).astype(int)
polycrystal = Polycrystal(mesh, orientation, strain=np.zeros(
    (3, 3)), phases=phases, element_phase_map=element_phase_map)

w = detector_size  # full field beam
beam_vertices = np.array([
    [-detector_distance, -w, -w],
    [-detector_distance, w, -w],
    [-detector_distance, w, w],
    [-detector_distance, -w, w],
    [detector_distance, -w, -w],
    [detector_distance, w, -w],
    [detector_distance, w, w],
    [detector_distance, -w, w]])
wavelength = 0.285227
xray_propagation_direction = np.array([1, 0, 0]) * 2 * np.pi / wavelength
polarization_vector = np.array([0, 1, 0])
beam = Beam(
    beam_vertices,
    xray_propagation_direction,
    wavelength,
    polarization_vector)

rotation_angle = 5.0 * np.pi / 180.
rotation_axis = np.array([0, 0, 1])
translation = np.array([0, 0, 0])
motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

print("Diffraction computations:")
pr = cProfile.Profile()
pr.enable()
peaks_dict = polycrystal.diffract(beam, detector, motion)
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(15)
print("")

# Print peak information from peaks_dict
import torch
peaks = peaks_dict["peaks"]
columns = peaks_dict["columns"]
if torch.is_tensor(peaks):
    peaks_np = peaks.cpu().numpy()
else:
    peaks_np = np.array(peaks)

h_idx = columns.index("h")
k_idx = columns.index("k")
l_idx = columns.index("l")
time_idx = columns.index("diffraction_times")
tth_idx = columns.index("2theta")

for i in range(min(5, len(peaks_np))):  # Print first 5 peaks
    hkl = peaks_np[i, h_idx:l_idx+1].astype(int)
    time = peaks_np[i, time_idx]
    tth = peaks_np[i, tth_idx]
    print(hkl, time)
    print(np.degrees(tth))
    print(" ")

print("Detector gauss rendering:")
pr = cProfile.Profile()
pr.enable()
diffraction_pattern1 = detector.render(
    peaks_dict,
    frames_to_render=1,
    method="gauss")
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(15)
print("")

print("Detector voigt rendering:")
pr = cProfile.Profile()
pr.enable()
diffraction_pattern2 = detector.render(
    peaks_dict,
    frames_to_render=1,
    method='voigt')
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(15)

# Convert to numpy for plotting
if hasattr(diffraction_pattern1, 'cpu'):
    diffraction_pattern1_np = diffraction_pattern1[0].cpu().numpy()
    diffraction_pattern2_np = diffraction_pattern2[0].cpu().numpy()
else:
    diffraction_pattern1_np = np.array(diffraction_pattern1[0])
    diffraction_pattern2_np = np.array(diffraction_pattern2[0])

# diffraction_pattern_np[ diffraction_pattern_np<=0 ] = 1
# diffraction_pattern_np = np.log(diffraction_pattern_np)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(diffraction_pattern1_np, cmap='gray')
ax[1].imshow(diffraction_pattern2_np, cmap='gray')
ax[0].set_title("Fast Gaussian rendering")
ax[1].set_title("Voigt profile rendering")
ax[0].set_xlabel("Peaks: " + str(len(peaks_dict['peaks'])))
ax[1].set_xlabel("Peaks: " + str(len(peaks_dict['peaks'])))
plt.show()
