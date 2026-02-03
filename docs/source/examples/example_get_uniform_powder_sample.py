"""Example: Create a uniform powder sample using templates.get_uniform_powder_sample.

This example demonstrates how to create a polycrystal sample with uniformly
distributed grain orientations, suitable for powder diffraction simulations.
"""
import numpy as np
from xrd_simulator import templates

# Define sample parameters
sample_bounding_radius = 100.0  # microns
number_of_grains = 500  # Number of randomly oriented grains

# Define phase parameters (quartz example)
unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]  # a, b, c, alpha, beta, gamma
sgname = 'P3221'  # Space group name

# Optional: strain tensor (default is zero strain)
strain_tensor = np.zeros((3, 3))

# Create the polycrystal sample
polycrystal = templates.get_uniform_powder_sample(
    sample_bounding_radius=sample_bounding_radius,
    number_of_grains=number_of_grains,
    unit_cell=unit_cell,
    sgname=sgname,
    strain_tensor=strain_tensor,
    path_to_cif_file=None,  # Optional: path to CIF file for structure factors
)

print(f"Created polycrystal with {polycrystal.mesh_sample.number_of_elements} elements")
print(f"Number of phases: {len(polycrystal.phases)}")
