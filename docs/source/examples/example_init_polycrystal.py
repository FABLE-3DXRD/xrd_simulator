import numpy as np
from scipy.spatial.transform import Rotation as R
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.polycrystal import Polycrystal

# The toplogy of the polycrystal is described by a tetrahedral mesh,
# xrd_simulator supports several ways to generate a mesh, here we
# generate meshed solid sphere using a level set.
mesh = TetraMesh.generate_mesh_from_levelset(
    level_set=lambda x: np.linalg.norm(x) - 768.0,
    bounding_radius=769.0,
    max_cell_circumradius=450.)

# Each element of the mesh is a single crystal with properties defined
# by an xrd_simulator.phase.Phase object.
quartz = Phase(unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
               sgname='P3221',  # (Quartz)
               path_to_cif_file=None  # phases can be defined from crystalographic information files
               )

# The polycrystal can now map phase(s) (only quartz here), orientations and
# strains to the tetrahedral mesh elements.
orientation = R.random(mesh.number_of_elements).as_matrix()
polycrystal = Polycrystal(mesh,
                          orientation,
                          strain=np.zeros((3, 3)),
                          phases=quartz,
                          element_phase_map=None)

# The polycrystal may be saved to disc for later usage.
polycrystal.save('my_polycrystal', save_mesh_as_xdmf=True)
polycrystal_loaded_from_disc = Polycrystal.load('my_polycrystal.pc')
