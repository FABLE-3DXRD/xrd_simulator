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

# The mesh can be saved as a .xdmf for visualization.
mesh.save(r'C:\Users\Henningsson\Downloads\mesh.xdmf')

# Each element of the mesh is a single crystal with properties defined
# by an xrd_simulator.phase.Phase object.
quartz = Phase(
    unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
    sgname='P3221'  # (Quartz)
)

# The polycrystal can now map phase(s) (only quartz here), orientations and
# strains to the tetrahedral mesh elements.
polycrystal = Polycrystal(mesh,
                          ephase=np.zeros((mesh.number_of_elements,)).astype(int),
                          crystal_orientations=R.random(mesh.number_of_elements).as_matrix(),
                          crystal_strain=np.zeros((mesh.number_of_elements,3,3)),
                          phases=[quartz])
