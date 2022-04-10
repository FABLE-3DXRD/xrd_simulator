import numpy as np
from xrd_simulator.mesh import TetraMesh


nodal_coordinates = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,-1]])
element_node_map  = np.array([[0,1,2,3],[0,1,2,4]])
# Generate meshed solid sphere using a level set.
mesh = TetraMesh.generate_mesh_from_vertices(nodal_coordinates, element_node_map)

# The mesh may be saved to disc for later usage or visualization.
mesh.save('my_mesh')
mesh_loaded_from_disc = mesh.load('my_mesh.xdmf')
