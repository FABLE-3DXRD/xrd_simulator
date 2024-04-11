import matplotlib.pyplot as plt
import numpy as np
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.templates import get_uniform_powder_sample
import os

pixel_size = 75.
detector_size = pixel_size * 1024
detector_distance = 142938.28756189224
det_corner_0 = np.array([detector_distance, -detector_size / 2., -detector_size / 2.])
det_corner_1 = np.array([detector_distance, detector_size / 2., -detector_size / 2.])
det_corner_2 = np.array([detector_distance, -detector_size / 2., detector_size / 2.])

detector = Detector(pixel_size, pixel_size, det_corner_0, det_corner_1, det_corner_2)
sample_bounding_radius = 0.0001 * detector_size
polycrystal = get_uniform_powder_sample(
    sample_bounding_radius=sample_bounding_radius,
    number_of_grains=500,
    unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
    sgname='P3221'
)


w = 2 * sample_bounding_radius  # full field beam
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

rotation_angle = 5 * np.pi / 180.
rotation_axis = np.array([0, 1, 0])
translation = np.array([0, 0, 0])
motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

polycrystal.diffract(beam, detector, motion)
diffraction_pattern = detector.render(frames_to_render=0, method='project')

plt.imshow(diffraction_pattern > 0, cmap='gray')
plt.title("Hits: " + str(len(detector.frames[0])))
plt.show()
