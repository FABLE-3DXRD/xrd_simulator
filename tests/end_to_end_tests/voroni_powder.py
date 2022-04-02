import matplotlib.pyplot as plt
import meshio
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

centroids = []
for i in range(2, 66):
    path = os.path.join(
        os.path.dirname(__file__),
        '../../extras/voroni_sample/grain_meshes')
    path = os.path.normpath(path)
    grainmeshfile = os.path.join(path, 'grain' + str(i).zfill(4) + '.xdmf')
    mesh = meshio.read(grainmeshfile)
    centroids.append(list((1 / 256.) * np.mean(mesh.points, axis=0)))
centroids = np.array(centroids)
sample_diameter = 1.0
coord, enod = [], []
k = 0
for c in centroids:
    coord.append([c[0], c[1], c[2]])
    coord.append([c[0] + 0.01, c[1], c[2]])
    coord.append([c[0], c[1] + 0.01, c[2]])
    coord.append([c[0], c[1], c[2] + 0.01])
    enod.append([k, k + 1, k + 2, k + 3])
    k += 3
coord = np.array(coord)
enod = np.array(enod)
mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)
print(mesh.coord)

print("")
print('nelm:', mesh.number_of_elements)
print("")

pixel_size = 5 * sample_diameter / 256.
detector_size = pixel_size * 1024
detector_distance = 50 * sample_diameter
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
    np.random.normal(loc=0.0, scale=np.radians(20), size=(mesh.number_of_elements, 3))
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
wavelength = 0.115227
xray_propagation_direction = np.array([1, 0, 0]) * 2 * np.pi / wavelength
polarization_vector = np.array([0, 1, 0])
beam = Beam(
    beam_vertices,
    xray_propagation_direction,
    wavelength,
    polarization_vector)

rotation_angle = 90.0 * np.pi / 180.
rotation_axis = np.array([0, 0, 1])
translation = np.array([0, 0, 0])
motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

print("Diffraction computations:")
pr = cProfile.Profile()
pr.enable()
polycrystal.diffract(beam, detector, motion)
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(10)
print("")

print("Detector centroid rendering:")
pr = cProfile.Profile()
pr.enable()
diffraction_pattern1 = detector.render(
    frame_number=0,
    lorentz=False,
    polarization=False,
    structure_factor=False,
    method="centroid")
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(10)
print("")

print("Detector project rendering:")
pr = cProfile.Profile()
pr.enable()
diffraction_pattern2 = detector.render(
    frame_number=0,
    lorentz=False,
    polarization=False,
    structure_factor=False,
    method='project')
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(10)

# diffraction_pattern[ diffraction_pattern<=0 ] = 1
# diffraction_pattern = np.log(diffraction_pattern)

# from scipy.signal import convolve

# kernel = np.ones((4,4))
# diffraction_pattern1 = convolve(diffraction_pattern1, kernel, mode='full', method='auto')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(diffraction_pattern1, cmap='gray')
ax[1].imshow(diffraction_pattern2, cmap='gray')
ax[0].set_title("Fast delta peak rendering")
ax[1].set_title("Full projection rendering")
ax[0].set_xlabel("Hits: " + str(len(detector.frames[0])))
ax[1].set_xlabel("Hits: " + str(len(detector.frames[0])))
plt.show()
