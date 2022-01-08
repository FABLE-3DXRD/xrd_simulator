import numpy as np
from xrd_simulator import templates

polycrystal = templates.get_uniform_powder_sample(
    sample_bounding_radius=1.203,
    number_of_grains=15,
    unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
    sgname='P3221',
    strain_tensor=np.array([[0, 0, -0.0012],
                            [0, 0, 0],
                            [0, 0, 0]])
)
