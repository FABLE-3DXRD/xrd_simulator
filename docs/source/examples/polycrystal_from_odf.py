import numpy as np
from xrd_simulator.templates import polycrystal_from_odf

ODF = lambda x,q: 1./(np.pi**2) # uniform orinetation distribution function.
number_of_crystals = 500
bounding_height = 50
bounding__radius = 25
unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
sgname = 'P3221', # Quartz
max_bin = np.radians(10.0)

polycrystal = polycrystal_from_odf( ODF,
                                    number_of_crystals,
                                    bounding_height,
                                    bounding__radius,
                                    unit_cell,
                                    sgname,
                                    max_bin )