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
import pygalmesh
import meshio
from xrd_simulator import utils
from xrd_simulator import motion


class TetraMesh(object):
    """Defines a 3D tetrahedral mesh with associated geometry data such face normals, centroids, etc.

    For level-set mesh generation the TetraMesh uses `the meshio package`_: For more meshing tools
    please see this package directly (which itself is a wrapper of CGAL)

     .. _the meshio package: https://github.com/nschloe/meshio

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
        """Generate a mesh from a level set using `the pygalmesh package`_:

        .. _the pygalmesh package: https://github.com/nschloe/pygalmesh

        Args:
            level_set (:obj:`callable`): Level set, level_set(x) should give a negative output on the exterior
                of the mesh and positive on the interior.
            bounding_radius (:obj:`float`): Bounding radius of mesh.
            max_cell_circumradius (:obj:`float`): Bound for element radii.

        """

        class LevelSet(pygalmesh.DomainBase):
            def __init__(self):
                super().__init__()
                self.eval = level_set
                self.get_bounding_sphere_squared_radius = lambda: bounding_radius**2

        mesh = pygalmesh.generate_mesh(
            LevelSet(), max_cell_circumradius=max_cell_circumradius, verbose=False
        )

        return cls._build_tetramesh(mesh)

    def translate(self, translation_vector):
        """Translate the mesh.

        Args:
            translation_vector (:obj:`numpy.array`): [x,y,z] translation vector, shape=(3,)

        """
        self._mesh.points += translation_vector
        self.coord = np.array(self._mesh.points)
        self.ecentroids += translation_vector
        self.espherecentroids += translation_vector
        self.centroid += translation_vector

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
        self._mesh.points = rigid_body_motion(self._mesh.points, time=time)
        self.coord = np.array(self._mesh.points)

        s1, s2, s3 = self.enormals.shape
        self.enormals = self.enormals.reshape(s1 * s2, 3)

        self.enormals = rigid_body_motion.rotate(self.enormals, time=time)
        self.enormals = self.enormals.reshape(s1, s2, s3)

        self.ecentroids = rigid_body_motion(self.ecentroids, time=time)
        self.espherecentroids = rigid_body_motion(self.espherecentroids, time=time)

        self.centroid = rigid_body_motion(self.centroid.reshape(1, 3), time=time)[0]

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
                element_data[key] = [list(element_data[key])]

        meshio.write_points_cells(
            save_path, self.coord, [("tetra", self.enod)], cell_data=element_data
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
        tetmesh = cls()
        tetmesh._mesh = mesh
        tetmesh._set_fem_matrices()
        tetmesh._expand_mesh_data()
        return tetmesh

    def _compute_mesh_faces(self, enod):
        """Compute all element faces nodal numbers. We create a matrix of all possible permutations and then we index the enod matrix."""
        permutations = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        efaces = enod[:, permutations]
        return efaces

    def _compute_mesh_normals(self, coord, enod, efaces):
        """Compute all element faces outwards unit vector normals."""
        vertices = coord[efaces]
        normals = np.cross(
            vertices[:, :, 1, :] - vertices[:, :, 0, :],
            vertices[:, :, 2, :] - vertices[:, :, 0, :],
        )
        faces_centers = np.mean(vertices, axis=2)
        centroids = np.mean(faces_centers, axis=1)
        centroid_to_face = faces_centers - centroids[:, np.newaxis, :]
        signs = np.sum(centroid_to_face * normals, axis=-1)
        signs[signs >= 0] = 1
        signs[signs < 0] = -1
        enormals = (
            normals
            / np.linalg.norm(normals, axis=2, keepdims=True)
            * signs[:, :, np.newaxis]
        )

        return enormals

    def _compute_mesh_centroids(self, coord, enod):
        """Compute centroids of elements."""
        ecentroids = np.mean(coord[enod], axis=1)
        return ecentroids

    def _compute_mesh_volumes(self, enod, coord):
        """Compute per element enclosed volume."""
        vertices = coord[enod]
        a = vertices[:, 1, :] - vertices[:, 0, :]
        b = vertices[:, 2, :] - vertices[:, 0, :]
        c = vertices[:, 3, :] - vertices[:, 0, :]
        evolumes = (1 / 6.0) * np.sum((np.cross(a, b, axis=1) * c), axis=1)
        return evolumes

    def _compute_mesh_spheres(self, coord, enod):
        """Compute per tetrahedron minimal bounding spheres. The approach here described avoids any iterative process,
        solving in a vectorized manner all the spheres at once.

        First the minimal sphere for the two most distant vertices of every tetrahedron is calculated.
        Then the minimal sphere for the three most distant vertices is calculated.
        Finally the sphere containing the 4 vertices in their surface is calculated.

        The first of the three spheres that satisfies all vertices of each tetrahedron being contained in it
        is selected.

        coord (coordinates) dimensions --> (triangle,vertices,xyz)
        enod (ids of triangles) dimensions --> (tetrahedron,faces)
        """
        vertices = coord[enod]
        n_tetra = enod.shape[0]
        range_n_tetra = range(n_tetra)
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        all_pairs = np.tile([0, 1, 2, 3], (n_tetra, 1))

        # We compute the length of every tetrahedron side
        sides = utils._compute_sides(vertices)

        # We compute the vertices that are further apart
        max_sides_index = np.argmax(sides, axis=1)
        furthest2_indices = pairs[max_sides_index]

        # We compute the other 2 vertices of the tetrahedra
        mask = np.zeros((n_tetra, 4), dtype=bool)
        mask[range_n_tetra, furthest2_indices[:, 0]] = 1
        mask[range_n_tetra, furthest2_indices[:, 1]] = 1
        other2_indices = all_pairs[~mask].reshape(-1, 2)

        # We compute the smallest spheres that pass through the farthest-apart vertices and check if the other-2 vertices fall in
        furthest2_vertices = vertices.transpose(1, 0, 2)[
            furthest2_indices.transpose(1, 0), range_n_tetra
        ].transpose(1, 0, 2)
        other2_vertices = vertices.transpose(1, 0, 2)[
            other2_indices.transpose(1, 0), range_n_tetra
        ].transpose(1, 0, 2)
        centers_1D, radii_1D = utils._circumsphere_of_segments(furthest2_vertices)
        dist_centers_1D_to_vertices = np.linalg.norm(
            other2_vertices - centers_1D[:, np.newaxis, :], axis=2
        )
        spheres_solved_with_1D = np.all(
            (dist_centers_1D_to_vertices - radii_1D[:, np.newaxis]) <= 0, axis=1
        )

        # We compute the smallest spheres that pass through the farthest 3 vertices and check if the remaining vertex falls in
        next_furthest_index = np.argmax(dist_centers_1D_to_vertices, axis=1)
        third_vertex = other2_vertices.transpose(1, 0, 2)[
            next_furthest_index, range_n_tetra
        ]
        largest_triangles = np.append(
            furthest2_vertices, third_vertex[:, np.newaxis, :], axis=1
        )
        centers_2D, radii_2D = utils._circumsphere_of_triangles(largest_triangles)
        dist_centers_2D_to_vertices = np.linalg.norm(
            vertices - centers_2D[:, np.newaxis, :], axis=2
        )
        spheres_solved_with_2D = np.all(
            (dist_centers_2D_to_vertices - radii_2D[:, np.newaxis]) <= 0, axis=1
        )

        # Finally we compute the spheres that contain the 4 points on the surfaces for the remaining tetrahedrons
        centers_3D, radii_3D = utils._circumsphere_of_tetrahedrons(vertices)

        # We select the 1D case if possible, otherwise the 2D case, and finally the 3D case
        espherecentroids = np.where(
            spheres_solved_with_1D[:, np.newaxis],
            centers_1D,
            np.where(spheres_solved_with_2D[:, np.newaxis], centers_2D, centers_3D),
        )
        eradius = np.where(
            spheres_solved_with_1D,
            radii_1D,
            np.where(spheres_solved_with_2D, radii_2D, radii_3D),
        )

        return eradius, espherecentroids

    def _set_fem_matrices(self):
        """Extract and set mesh FEM matrices from pygalmesh object."""
        self.coord = np.array(self._mesh.points)
        self.enod = np.array(self._mesh.cells_dict["tetra"])
        self.dof = np.arange(0, self.coord.shape[0] * 3).reshape(self.coord.shape[0], 3)
        self.number_of_elements = self.enod.shape[0]

    def _expand_mesh_data(self):
        """Compute extended mesh quantities such as element faces and normals."""
        self.efaces = self._compute_mesh_faces(self.enod)
        self.enormals = self._compute_mesh_normals(self.coord, self.enod, self.efaces)
        self.ecentroids = self._compute_mesh_centroids(self.coord, self.enod)
        self.eradius, self.espherecentroids = self._compute_mesh_spheres(
            self.coord, self.enod
        )
        self.centroid = np.mean(self.ecentroids, axis=0)
        self.evolumes = self._compute_mesh_volumes(self.enod, self.coord)

        # TODO: considering leveraging this in beam.py for speed
