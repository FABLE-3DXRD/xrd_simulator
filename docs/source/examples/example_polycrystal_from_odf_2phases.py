import numpy as np
from xrd_simulator.templates import polycrystal_from_odf


# uniform orientation distribution function.
def ODF(x, q): return 1. / (np.pi**2)


number_of_crystals = 500
bounding_height = 50.0
bounding_radius = 25.0
unit_cell = [[3.579, 3.579, 3.579, 90., 90., 90.0],
            [5.46745, 5.46745, 5.46745, 90., 90., 90.0]]
sgname = ['F432', 'F432']
max_bin = np.radians(10.0)
path_to_cif_file = None

def strain_tensor(x): return np.array([[0, 0, 0.02 * x[2] / bounding_height],
                                       [0, 0, 0],
                                       [0, 0, 0]])  # Linear strain gradient along rotation axis.


polycrystal = polycrystal_from_odf(ODF,
                                   number_of_crystals,
                                   bounding_height,
                                   bounding_radius,
                                   unit_cell,
                                   sgname,
                                   path_to_cif_file,
                                   max_bin,
                                   strain_tensor)
