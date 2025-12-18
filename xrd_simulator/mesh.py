"""The mesh module is used to represent the morphology of a polycrystalline sample.
Once created and linked to a polycrystal the mesh can be accessed directly through
the :class:`xrd_simulator.polycrystal.Polycrystal`. Here is a minimal example of how
to instantiate a mesh and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_mesh.py

This should look somethign like this in a 3D viewer like paraview:

.. image:: https://github.com/FABLE-3DXRD/xrd_simulator/blob/main/docs/source/images/mesh_example.png?raw=true
   :width: 300
   :align: center

Below follows a detailed description of the mesh class attributes and functions.

"""

import numpy as np
import meshpy.tet as tet
import meshio
import torch
from xrd_simulator import utils, motion


class TetraMesh(object):
    """Defines a 3D tetrahedral mesh with associated geometry data such face normals, centroids, etc.

    For level-set mesh generation the TetraMesh uses marching cubes from `scikit-image`_ to extract
    the surface, then `meshpy`_ (TetGen wrapper) to generate the volume tetrahedral mesh.

     .. _scikit-image: https://scikit-image.org/
     .. _meshpy: https://github.com/inducer/meshpy

    Attributes:
        coord (:obj:`numpy array`): Nodal coordinates, shape=(nenodes, 3). Each row in coord defines the
            coordinates of a mesh node.
        enod (:obj:`numpy array`): Tetra element nodes shape=(nelm, nenodes).e.g enod[i,:] gives
            the nodal indices of element i.
        dof (:obj:`numpy array`): Per node degrees of freedom, i.e dof[i,:]
            gives the degrees of freedom of node i.
        efaces (:obj:`numpy array`): Element faces nodal indices, shape=(nelm, nenodes, 3).
            e.g efaces[i,j,:] gives the nodal indices of face j of element i.
        enormals (:obj:`numpy array`): Element faces outwards normals (nelm, nefaces, 3).
            e.g enormals[i,j,:] gives the normal of face j of element i.
        ecentroids (:obj:`numpy array`): Per element centroids, shape=(nelm, 3).
        eradius (:obj:`numpy array`): Per element bounding ball radius, shape=(nelm, 1).
        espherecentroids (:obj:`numpy array`): Per element bounding ball centroids, shape=(nelm, 3).
        evolumes (:obj:`numpy array`): Per element volume, shape=(nelm,).
        centroid (:obj:`numpy array`): Global centroid of the entire mesh, shape=(3,)
        number_of_elements (:obj:`int`): Number of tetrahedral elements in the mesh.

    """

    def __init__(self):
        self._mesh = None
        self.coord = None
        self.enod = None
        self.dof = None
        self.efaces = None
        self.enormals = None
        self.ecentroids = None
        self.eradius = None
        self.espherecentroids = None
        self.evolumes = None
        self.centroid = None
        self.number_of_elements = None

    @classmethod
    def generate_mesh_from_vertices(cls, coord, enod):
        """Generate a mesh from vertices using `the meshio package`_:

        .. _the meshio package: https://github.com/nschloe/meshio

        Args:
            coord (:obj:`numpy array`): Nodal coordinates, shape=(nenodes, 3). Each row in coord defines the
                coordinates of a mesh node.
            enod (:obj:`numpy array`): Tetra element nodes shape=(nelm, nenodes).e.g enod[i,:] gives
                the nodal indices of element i.

        """
        mesh = meshio.Mesh(coord, [("tetra", enod)])
        return cls._build_tetramesh(mesh)

    @classmethod
    def generate_mesh_from_levelset(
        cls, level_set, bounding_radius, max_cell_circumradius
    ):
        """Generate a mesh from a level set using marching cubes (scikit-image) + meshpy.

        This method uses a two-step process:
        1. Extract the surface from the level set using marching cubes (skimage)
        2. Generate a volume tetrahedral mesh from the surface using meshpy

        Args:
            level_set (:obj:`callable`): Level set, level_set(x) should give a negative output on the exterior
                of the mesh and positive on the interior.
            bounding_radius (:obj:`float`): Bounding radius of mesh.
            max_cell_circumradius (:obj:`float`): Bound for element radii.

        Returns:
            TetraMesh: The generated tetrahedral mesh
        """
        from skimage import measure
        
        # Create a grid to sample the level set
        # Use a reasonably fine grid to capture the surface accurately
        n = max(int(2 * bounding_radius / (max_cell_circumradius * 0.5)), 20)
        x = np.linspace(-bounding_radius, bounding_radius, n, dtype=np.float64)
        y = np.linspace(-bounding_radius, bounding_radius, n, dtype=np.float64)
        z = np.linspace(-bounding_radius, bounding_radius, n, dtype=np.float64)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Evaluate level set function on the grid
        points_grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        values = np.array([level_set(p) for p in points_grid], dtype=np.float64)
        volume = values.reshape(X.shape)
        
        # Extract the surface mesh using marching cubes at level=0
        # Note: marching_cubes extracts the surface where volume == level
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume, level=0.0, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
            )
        except (ValueError, RuntimeError) as e:
            raise ValueError(
                f"Marching cubes failed to extract surface: {str(e)}. "
                "This likely means the level set doesn't intersect the zero level "
                "within the bounding box, or doesn't define a valid closed surface."
            )
        
        # Transform vertices from grid coordinates to world coordinates and ensure float64
        verts = verts.astype(np.float64) + np.array([x[0], y[0], z[0]], dtype=np.float64)
        faces = faces.astype(np.int32)  # Ensure integer type for faces
        
        # Verify we have a valid surface
        if len(verts) == 0 or len(faces) == 0:
            raise ValueError(
                "Marching cubes produced no surface. "
                "Check that the level set crosses zero within the bounding_radius."
            )
        
        # Create mesh info for meshpy
        mesh_info = tet.MeshInfo()
        mesh_info.set_points(verts.tolist())
        
        # Set the surface facets - these define the boundary
        mesh_info.set_facets(faces.tolist())
        
        # Build the volume mesh with quality constraints
        max_volume = (max_cell_circumradius ** 3) / 6.0
        
        try:
            mesh = tet.build(
                mesh_info,
                max_volume=max_volume,
                attributes=True
            )
        except Exception as e:
            raise ValueError(
                f"Tetrahedral mesh generation failed: {str(e)}. "
                "The surface may be self-intersecting or have other topological issues. "
                "Try adjusting bounding_radius or max_cell_circumradius."
            )
        
        # Convert to meshio format with explicit double precision
        vertices = np.array(mesh.points, dtype=np.float64)
        elements = np.array(mesh.elements, dtype=np.int64)
        
        # Verify mesh generation succeeded
        if len(vertices) == 0 or len(elements) == 0:
            raise ValueError(
                "Mesh generation produced no tetrahedra. "
                "This may indicate an issue with the surface topology."
            )
        
        mesh_obj = meshio.Mesh(vertices, [("tetra", elements)])
        
        return cls._build_tetramesh(mesh_obj)

    def translate(self, translation_vector):
        """Translate the mesh.

        Args:
            translation_vector (:obj:`numpy.array` or :obj:`torch.Tensor`): [x,y,z] translation vector, shape=(3,)

        """
        # Convert to numpy if torch tensor
        if torch.is_tensor(translation_vector):
            translation_vector = utils.ensure_numpy(translation_vector)
        
        self._mesh.points += translation_vector
        self.coord = utils.ensure_torch(self._mesh.points, dtype=torch.float64)
        self.ecentroids += utils.ensure_torch(translation_vector, dtype=torch.float64)
        self.espherecentroids += utils.ensure_torch(translation_vector, dtype=torch.float64)
        self.centroid += utils.ensure_torch(translation_vector, dtype=torch.float64)

    def rotate(self, rotation_axis, angle):
        """Rotate the mesh.

        Args:
            rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
            rotation_angle (:obj:`float`): Radians for rotation.

        """
        rbm = motion.RigidBodyMotion(rotation_axis, angle, np.array([0, 0, 0]))
        self.update(rbm, time=1)

    def update(self, rigid_body_motion, time):
        """Apply a rigid body motion transformation to the mesh.

        Args:
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].
            time (:obj:`float`): Time between [0,1] at which to call the rigid body motion.

        """
        self.coord = rigid_body_motion(self.coord, time=time)

        # Reshape and rotate normals
        s1, s2, s3 = self.enormals.shape
        self.enormals = self.enormals.reshape(s1 * s2, 3)
        self.enormals = rigid_body_motion.rotate(self.enormals, time=time)
        self.enormals = self.enormals.reshape(s1, s2, s3)

        # Update centroids and sphere data
        self.ecentroids = rigid_body_motion(self.ecentroids, time=time)
        self.espherecentroids = rigid_body_motion(self.espherecentroids, time=time)
        self.centroid = rigid_body_motion(self.centroid.unsqueeze(0), time=time).squeeze(0)

    def save(self, file, element_data=None):
        """Save the tetra mesh to .xdmf paraview readable format for visualization.

        Args:
            file (:obj:`str`): Absolute path to save the mesh in .xdmf format.
            element_data (:obj:`dict` of :obj:`list` or :obj:`numpy array`): Data associated to the elements.

        """
        if not file.endswith(".xdmf"):
            save_path = file + ".xdmf"
        else:
            save_path = file

        if element_data is not None:
            for key in element_data:
                if torch.is_tensor(element_data[key]):
                    element_data[key] = [list(utils.ensure_numpy(element_data[key]))]
                else:
                    element_data[key] = [list(element_data[key])]
        
        # Convert coord and enod to numpy if they're torch tensors
        coord = utils.ensure_numpy(self.coord) if torch.is_tensor(self.coord) else self.coord
        enod = utils.ensure_numpy(self.enod) if torch.is_tensor(self.enod) else self.enod

        meshio.write_points_cells(
            save_path, coord, [("tetra", enod)], cell_data=element_data
        )

    @classmethod
    def load(cls, path):
        """Load a mesh from a saved mesh file set using `the meshio package`_:

        .. _the meshio package: https://github.com/nschloe/meshio

        Args:
            file (:obj:`str`): Absolute path to the mesh file.

        """
        mesh = meshio.read(path)
        return cls._build_tetramesh(mesh)

    @classmethod
    def _build_tetramesh(cls, mesh):
        """Build a TetraMesh object from a meshio mesh, converting data to tensors immediately.
        
        Args:
            mesh: A meshio.Mesh object with points and tetra cells.
            
        Returns:
            TetraMesh: A new mesh object with all data as torch tensors.
        """
        tetmesh = cls()
        
        # Convert core mesh data to tensors immediately
        tetmesh.coord = utils.ensure_torch(mesh.points, dtype=torch.float64)
        
        # Handle enod - ensure it's 2D
        enod_data = mesh.cells_dict["tetra"]
        enod_tensor = utils.ensure_torch(enod_data, dtype=torch.int64)
        # If enod is 1D (single tetrahedron), reshape to 2D
        if enod_tensor.ndim == 1:
            enod_tensor = enod_tensor.reshape(1, -1)
        tetmesh.enod = enod_tensor
        
        # Store tensor versions in _mesh for persistence
        tetmesh._mesh = meshio.Mesh(
            points=utils.ensure_numpy(tetmesh.coord),
            cells=[("tetra", utils.ensure_numpy(tetmesh.enod))]
        )
        
        # Complete initialization using tensor data
        tetmesh._set_fem_matrices()
        tetmesh._expand_mesh_data()
        return tetmesh

    def _compute_mesh_faces(self, enod):
        """Compute all element faces nodal numbers. We create a matrix of all possible permutations and then we index the enod matrix."""
        # Ensure input enod is long tensor
        if not torch.is_tensor(enod):
            enod = torch.tensor(enod, dtype=torch.int64)
        elif enod.dtype != torch.int64:
            enod = enod.to(dtype=torch.int64)
            
        # Create permutations as long tensor
        permutations = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], 
                                  dtype=torch.int64, device=enod.device)
        efaces = enod[:, permutations]
        return efaces

    def _compute_mesh_normals(self, coord, enod, efaces):
        """Compute all element faces outwards unit vector normals."""
        vertices = coord[efaces]
        normals = torch.cross(
            vertices[:, :, 1, :] - vertices[:, :, 0, :],
            vertices[:, :, 2, :] - vertices[:, :, 0, :],
            dim=2
        )
        faces_centers = torch.mean(vertices, dim=2)
        centroids = torch.mean(faces_centers, dim=1)
        centroid_to_face = faces_centers - centroids.unsqueeze(1)
        signs = torch.sum(centroid_to_face * normals, dim=-1)
        signs = torch.where(signs >= 0, torch.tensor(1.0, device=signs.device), torch.tensor(-1.0, device=signs.device))
        enormals = normals / torch.linalg.norm(normals, dim=2, keepdim=True) * signs.unsqueeze(-1)
        return enormals

    def _compute_mesh_centroids(self, coord, enod):
        """Compute centroids of elements."""
        ecentroids = torch.mean(coord[enod], dim=1)
        return ecentroids

    def _compute_mesh_volumes(self, enod, coord):
        """Compute per element enclosed volume."""
        vertices = coord[enod]
        a = vertices[:, 1, :] - vertices[:, 0, :]
        b = vertices[:, 2, :] - vertices[:, 0, :]
        c = vertices[:, 3, :] - vertices[:, 0, :]
        evolumes = (1 / 6.0) * torch.sum(torch.cross(a, b, dim=1) * c, dim=1)
        return evolumes

    def _compute_mesh_spheres(self, coord, enod):
        """
        Compute the minimal bounding spheres for each tetrahedron in a mesh. This approach avoids any iterative process 
        and solves all the spheres in a vectorized manner at once.

        The method follows these steps:
        1. Calculate the minimal sphere for the two most distant vertices of each tetrahedron.
        2. Calculate the minimal sphere for the three most distant vertices.
        3. Calculate the sphere containing all four vertices on their surface.

        Parameters:
        ----------
        coord : torch.Tensor
            Tensor of coordinates with dimensions (vertices, xyz).
        enod : torch.Tensor
            Tensor of tetrahedron vertex indices with dimensions (tetrahedrons, vertices).

        Returns:
        -------
        eradius : torch.Tensor
            Tensor of radii for the minimal bounding spheres of each tetrahedron.
        espherecentroids : torch.Tensor
            Tensor of centroids for the minimal bounding spheres of each tetrahedron.
        """
        # Ensure inputs are torch tensors
        coord = utils.ensure_torch(coord, dtype=torch.float64)
        enod = utils.ensure_torch(enod, dtype=torch.int64)
        
        vertices = coord[enod]
        n_tetra = enod.shape[0]
        pairs = torch.tensor(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            dtype=torch.int64,
            device=coord.device,
        )
        all_pairs = torch.tile(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=coord.device),
            (n_tetra, 1),
        )

        # Compute length of every tetrahedron side
        sides = utils._compute_sides(vertices)

        # Find vertices that are furthest apart
        max_sides_index = torch.argmax(sides, dim=1)
        furthest2_indices = pairs[max_sides_index]

        # Compute other 2 vertices of the tetrahedra
        mask = torch.zeros((n_tetra, 4), dtype=torch.bool, device=coord.device)
        mask[torch.arange(n_tetra, device=coord.device), furthest2_indices[:, 0]] = True
        mask[torch.arange(n_tetra, device=coord.device), furthest2_indices[:, 1]] = True
        other2_indices = all_pairs[~mask].reshape(-1, 2)

        # Compute smallest spheres through farthest-apart vertices
        furthest2_vertices = vertices.permute(1, 0, 2)[
            furthest2_indices.T, torch.arange(n_tetra, device=coord.device)
        ].permute(1, 0, 2)
        other2_vertices = vertices.permute(1, 0, 2)[
            other2_indices.T, torch.arange(n_tetra, device=coord.device)
        ].permute(1, 0, 2)
        centers_1D, radii_1D = utils._circumsphere_of_segments(furthest2_vertices)
        dist_centers_1D_to_vertices = torch.linalg.norm(
            other2_vertices - centers_1D.unsqueeze(1), dim=2
        )
        spheres_solved_with_1D = torch.all(
            (dist_centers_1D_to_vertices - radii_1D.unsqueeze(1)) <= 0, dim=1
        )

        # Compute spheres through farthest 3 vertices
        next_furthest_index = torch.argmax(dist_centers_1D_to_vertices, dim=1)
        third_vertex = other2_vertices.permute(1, 0, 2)[
            next_furthest_index, torch.arange(n_tetra, device=coord.device)
        ]
        largest_triangles = torch.cat(
            [furthest2_vertices, third_vertex.unsqueeze(1)], dim=1
        )
        centers_2D, radii_2D = utils._circumsphere_of_triangles(largest_triangles)
        dist_centers_2D_to_vertices = torch.linalg.norm(
            vertices - centers_2D.unsqueeze(1), dim=2
        )
        spheres_solved_with_2D = torch.all(
            (dist_centers_2D_to_vertices - radii_2D.unsqueeze(1)) <= 0, dim=1
        )

        # Compute spheres containing all 4 points
        centers_3D, radii_3D = utils._circumsphere_of_tetrahedrons(vertices)

        # Select appropriate solution based on containment tests
        espherecentroids = torch.where(
            spheres_solved_with_1D.unsqueeze(1),
            centers_1D,
            torch.where(spheres_solved_with_2D.unsqueeze(1), centers_2D, centers_3D)
        )
        eradius = torch.where(
            spheres_solved_with_1D,
            radii_1D,
            torch.where(spheres_solved_with_2D, radii_2D, radii_3D)
        )

        return eradius, espherecentroids

    def _set_fem_matrices(self):
        """Extract and set mesh FEM matrices from meshio object and convert to torch tensors."""
        # Convert coordinates to float64 tensor
        self.coord = utils.ensure_torch(self._mesh.points, dtype=torch.float64)
        
        # Convert element indices to long (int64) tensor for indexing
        self.enod = utils.ensure_torch(self._mesh.cells_dict["tetra"], dtype=torch.int64)
        if self.enod.dtype != torch.int64:
            self.enod = self.enod.to(dtype=torch.int64)
            
        # Generate and convert DOF indices to long tensor
        self.dof = utils.ensure_torch(
            np.arange(0, self.coord.shape[0] * 3).reshape(self.coord.shape[0], 3), 
            dtype=torch.int64
        )
        
        self.number_of_elements = self.enod.shape[0]

    def _expand_mesh_data(self):
        """Compute extended mesh quantities such as element faces and normals."""
        # Compute mesh faces, ensuring long tensor indices
        self.efaces = self._compute_mesh_faces(self.enod)
        if self.efaces.dtype != torch.int64:
            self.efaces = self.efaces.to(dtype=torch.int64)
            
        # Compute mesh normals and other geometric properties
        self.enormals = utils.ensure_torch(
            self._compute_mesh_normals(self.coord, self.enod, self.efaces),
            dtype=torch.float64
        )
        self.ecentroids = utils.ensure_torch(
            self._compute_mesh_centroids(self.coord, self.enod),
            dtype=torch.float64
        )
        
        # Compute bounding spheres
        eradius_np, espherecentroids_np = self._compute_mesh_spheres(self.coord, self.enod)
        self.eradius = utils.ensure_torch(eradius_np, dtype=torch.float64)
        self.espherecentroids = utils.ensure_torch(espherecentroids_np, dtype=torch.float64)
        
        # Compute global properties
        self.centroid = torch.mean(self.ecentroids, dim=0)
        self.evolumes = utils.ensure_torch(
            self._compute_mesh_volumes(self.enod, self.coord),
            dtype=torch.float64
        )
