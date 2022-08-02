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
            cls,
            level_set,
            bounding_radius,
            max_cell_circumradius):
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
            LevelSet(),
            max_cell_circumradius=max_cell_circumradius,
            verbose=False)

        return cls._build_tetramesh(mesh)

    def update(self, rigid_body_motion, time):
        """Apply a rigid body motion transformation to the mesh.

        Args:
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].
            time (:obj:`float`): Time between [0,1] at which to call the rigid body motion.

        """
        self._mesh.points = rigid_body_motion(self._mesh.points.T, time=time).T
        self.coord = np.array(self._mesh.points)

        s1,s2,s3 = self.enormals.shape
        self.enormals = self.enormals.reshape(s1*s2, 3)
        self.enormals = rigid_body_motion.rotate( self.enormals.T, time=time).T
        self.enormals = self.enormals.reshape(s1, s2, s3)

        self.ecentroids = rigid_body_motion(self.ecentroids.T, time=time).T
        self.espherecentroids = rigid_body_motion(self.espherecentroids.T, time=time).T
        self.centroid = rigid_body_motion(self.centroid.reshape(3,1), time=time).reshape(3,)

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
            save_path, self.coord, [
                ("tetra", self.enod)], cell_data=element_data)

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
        """Compute all element faces nodal numbers.
        """
        efaces = np.zeros((enod.shape[0], 4, 3), dtype=int)
        for i in range(enod.shape[0]):
            # nodal combinations defining 4 unique planes in a tet.
            permutations = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            for j, perm in enumerate(permutations):
                efaces[i, j, :] = enod[i, perm]
        return efaces

    def _compute_mesh_normals(self, coord, enod, efaces):
        """Compute all element faces outwards unit vector normals.
        """
        enormals = np.zeros((enod.shape[0], 4, 3))
        for i in range(enod.shape[0]):
            ec = coord[enod[i, :], :]
            ecentroid = np.mean(ec, axis=0)
            for j in range(efaces.shape[1]):
                ef = coord[efaces[i, j, :], :]
                enormals[i, j, :] = self._compute_plane_normal(ef, ecentroid)
        return enormals

    def _compute_plane_normal(self, points, centroid):
        """Compute plane normal (outwards refering to centroid).
        """
        v1 = points[1, :] - points[0, :]
        v2 = points[2, :] - points[0, :]
        # define a vector perpendicular to the plane.
        n = np.cross(v1, v2)
        # set vector direction outwards from centroid.
        n = n * np.sign(n.dot(points[0, :] - centroid))
        # normalised vector and return.
        return n / np.linalg.norm(n)

    def _compute_mesh_centroids(self, coord, enod):
        """Compute centroids of elements.
        """
        ecentroids = np.zeros((enod.shape[0], 3))
        for i in range(enod.shape[0]):
            ec = coord[enod[i, :], :]
            ecentroids[i, :] = np.sum(ec, axis=0) / ec.shape[0]
        return ecentroids

    def _compute_mesh_volumes(self, enod, coord):
        """Compute per element enclosed volume.
        """
        evolumes = np.zeros((enod.shape[0],))
        for i in range(enod.shape[0]):
            ec = coord[enod[i, :], :]
            a = ec[1] - ec[0]
            b = ec[2] - ec[0]
            c = ec[3] - ec[0]
            evolumes[i] = (1 / 6.) * np.dot(np.cross(a, b), c)
        return evolumes

    def _compute_mesh_spheres(self, coord, enod):
        """Compute per element minimal bounding spheres.
        """
        eradius = np.zeros((enod.shape[0],))
        espherecentroids = np.zeros((enod.shape[0], 3))
        for i in range(enod.shape[0]):
            ec = coord[enod[i, :], :]
            espherecentroids[i], eradius[i] = utils._get_bounding_ball(ec)
        return eradius, espherecentroids

    def _set_fem_matrices(self):
        """Extract and set mesh FEM matrices from pygalmesh object.
        """
        self.coord = np.array(self._mesh.points)
        self.enod = np.array(self._mesh.cells_dict['tetra'])
        self.dof = np.arange(
            0,
            self.coord.shape[0] *
            3).reshape(
            self.coord.shape[0],
            3)
        self.number_of_elements = self.enod.shape[0]

    def _expand_mesh_data(self):
        """Compute extended mesh quantities such as element faces and normals.
        """
        self.efaces = self._compute_mesh_faces(self.enod)
        self.enormals = self._compute_mesh_normals(
            self.coord, self.enod, self.efaces)
        self.ecentroids = self._compute_mesh_centroids(self.coord, self.enod)
        self.eradius, self.espherecentroids = self._compute_mesh_spheres(
            self.coord, self.enod)
        self.centroid = np.mean(self.ecentroids, axis=0)
        self.evolumes = self._compute_mesh_volumes(self.enod, self.coord)

        # TODO: considering leveraging this in beam.py for speed
        # self.econvexhulls    = [ ConvexHull( self.coord[nodes,:] ) for nodes in self.enod ]
