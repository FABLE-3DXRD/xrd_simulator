import numpy as np
import os
from xrd_simulator.beam import Beam

# The beam of xrays is represented as a convex polyhedron
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

# The xray beam object is instantiated
beam = Beam(
    beam_vertices,
    xray_propagation_direction=np.array([1., 0., 0.]),
    wavelength=0.28523,
    polarization_vector=np.array([0., 1., 0.]))

# The xray beam may be saved to disc for later usage.
artifacts_dir = os.path.join(os.path.dirname(__file__), 'test_artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
beam.save(os.path.join(artifacts_dir, 'my_xray_beam'))
beam_loaded_from_disc = Beam.load(os.path.join(artifacts_dir, 'my_xray_beam.beam'))