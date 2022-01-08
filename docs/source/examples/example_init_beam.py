import numpy as np
from xrd_simulator.beam import Beam

# The beam of x-rays is represented as a convex polyhedron
# We specify the vertices in a numpy array.
beam_vertices = np.array([
    [-5., 0., 0.],
    [-5., 1., 0.],
    [-5., 0., 1.],
    [-5., 1., 1.],
    [5., 0., 0.],
    [5., 1., 0.],
    [5., 0., 1.],
    [5., 1., 1.]])

beam = Beam(
    beam_vertices,
    xray_propagation_direction=np.array([1., 0., 0.]),
    wavelength=0.28523,
    polarization_vector=np.array([0., 1., 0.]))
