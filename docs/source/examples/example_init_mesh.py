import numpy as np
import os
from xrd_simulator.mesh import TetraMesh

# Generate mesh with 2 elements from nodal coordinates.
nodal_coordinates = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,-1]])
element_node_map  = np.array([[0,1,2,3],[0,1,2,4]])
mesh = TetraMesh.generate_mesh_from_vertices(nodal_coordinates, element_node_map)

# The mesh may be saved to disc for later usage or visualization.
artifacts_dir = os.path.join(os.path.dirname(__file__), 'test_artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
mesh.save(os.path.join(artifacts_dir, 'my_mesh'))
mesh_loaded_from_disc = TetraMesh.load(os.path.join(artifacts_dir, 'my_mesh.xdmf'))
