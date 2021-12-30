import numpy as np
from xrd_simulator.templates import polycrystal_from_odf

ODF = lambda x,q: 1./(np.pi**2) # uniform orientation distribution function.
number_of_crystals = 500
bounding_height = 50.0
bounding_radius = 25.0
unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
sgname = 'P3221', # Quartz
max_bin = np.radians(10.0)
strain_tensor = lambda x: np.array([ [0,0, 0.02*x[2]/bounding_height],
                                     [0,0,0],
                                     [0,0,0] ] ) # Linear strain gradient along rotation axis.

polycrystal = polycrystal_from_odf( ODF,
                                    number_of_crystals,
                                    bounding_height,
                                    bounding_radius,
                                    unit_cell,
                                    sgname,
                                    max_bin,
                                    strain_tensor )