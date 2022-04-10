import numpy as np
from xrd_simulator.mesh import TetraMesh

# Generate mesh with 2 elements from nodal coordinates.
nodal_coordinates = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,-1]])
element_node_map  = np.array([[0,1,2,3],[0,1,2,4]])
mesh = TetraMesh.generate_mesh_from_vertices(nodal_coordinates, element_node_map)

# The mesh may be saved to disc for later usage or visualization.
mesh.save('my_mesh')
mesh_loaded_from_disc = mesh.load('my_mesh.xdmf')
