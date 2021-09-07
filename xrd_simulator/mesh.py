import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from numba import njit
import pygalmesh

@njit
def _get_candidate_elements( point, ecentroids, eradius ):
    """Find all elements that could contain a point based on their bounding radii.
    This is a just in time compiled helper function for :obj:`TetraMesh` used for fast interpolation.
    """
    distance_vectors = ecentroids - point
    euclidean_distances = np.sum( distance_vectors*distance_vectors, axis=1 )
    candidate_elements = np.where( euclidean_distances <= eradius**2 )[0] 
    return candidate_elements[ np.argsort( euclidean_distances[candidate_elements] ) ]   

@njit
def _is_in_element( element, point, efaces, enormals, coord ):
    """Check if a point is contained by an element.
    This is a just in time compiled helper function for :obj:`TetraMesh` used for fast interpolation.
    """
    for face in range(efaces.shape[1]):
        face_node = efaces[element, face, 0]
        if (coord[face_node,:] - point).dot( enormals[element,face,:] ) < 0:
            break
        if face==efaces.shape[1]-1:
            return True
    return False

@njit
def _find_element_owner( point, ecentroids, eradius, enormals, coord, efaces, enod, element_guess):
    """Find the element that contains a point.
    This is a just in time compiled helper function for :obj:`TetraMesh` used for  fast interpolation.
    """ 
    if _is_in_element(element_guess, point, efaces, enormals, coord):
        return element_guess

    candidate_elements  = _get_candidate_elements(point, ecentroids, eradius)
    for element in candidate_elements:
        if _is_in_element(element, point, efaces, enormals, coord):
            return element

    return -1

@njit
def _get_interpolation_values_nd( xs, ys, zs, ecentroids, eradius, enormals, coord, efaces, ecmat, enod, coefficents ):
    """Compute mesh interpolated vector values at a series of coordinates.
    This is a just in time compiled helper function for :obj:`TetraMesh` used for fast interpolation.
    """
    values = np.zeros((len(xs),3))
    element_guess = 0
    for i,(x,y,z) in enumerate(zip(xs, ys, zs)):
        element = _find_element_owner( np.array([x,y,z]), ecentroids, eradius, enormals, coord, efaces, enod, element_guess)
        if element<0: 
            values[i,:] = 0
        else:
            element_nodes = enod[element, :]
            element_coefficents = coefficents[element_nodes, :]
            values[i,:] = element_coefficents.T.dot(  ecmat[element,:,:].T.dot(np.array([1,x,y,z])) )
            element_guess = element
    return values

@njit
def _get_interpolation_values_1d( xs, ys, zs, ecentroids, eradius, enormals, coord, efaces, ecmat, enod, coefficents, dim ):
    """Compute mesh interpolated scalar values at a series of coordinates.
    This is a just in time compiled helper function for :obj:`TetraMesh` used for fast interpolation.
    """ 
    values = np.zeros((len(xs),))
    element_guess = 0
    for i,(x,y,z) in enumerate(zip(xs, ys, zs)):
        element = _find_element_owner( np.array([x,y,z]), ecentroids, eradius, enormals, coord, efaces, enod, element_guess)
        if element<0: 
            values[i] = 0
        else:
            element_nodes = enod[element, :]
            element_coefficents = coefficents[element_nodes, dim]
            values[i] = element_coefficents.T.dot(  ecmat[element,:,:].T.dot(np.array([1,x,y,z])) )
            element_guess = element
    return values

class TetraMesh(object):
    """Defines a 3D tetrahedral finite element type basis by subclassing :obj:`Basis`. 
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
        eradius (:obj:`numpy array`): Per element bounding radius, shape=(nelm, 1).
        ecmat (:obj:`numpy array`): Per element interpolation matrix, shape=(nelm, 4, 4). When
            multiplied on a coordinate array, :obj:`np.array([1,x,y,z])`, the interpolated value 
            at x,y,z is found, given that x,y,z is contained by the corresponding element. 
    """

    def __init__(self):
        super().__init__()
        self._mesh       = None
        self.coord       = None
        self.enod        = None
        self.dof         = None
        self.efaces      = None
        self.enormals    = None
        self.ecentroids  = None
        self.eradius     = None
        self.ecmat       = None

    @classmethod
    def generate_mesh_from_levelset(cls, level_set, bounding_radius, max_cell_circumradius, max_facet_distance):
        """Generate a mesh from a level set using `the pygalmesh package`_:
        
        .. _the pygalmesh package: https://github.com/nschloe/pygalmesh
        Args:
            level_set (:obj:`callable`): Level set, level_set(x) should give a negative output on the exterior
                of the mesh and positive on the interior.
            bounding_radius (:obj:`float`): Bounding radius of mesh.
            max_cell_circumradius (:obj:`float`): Bound for element radii.
            max_facet_distance (:obj:`float`): Bound for facet distance.
        """

        class LevelSet(pygalmesh.DomainBase):
            def __init__(self):
                super().__init__()
                self.eval = level_set
                self.get_bounding_sphere_squared_radius = lambda : bounding_radius**2

        mesh = pygalmesh.generate_mesh( LevelSet(),
                                        max_facet_distance=max_facet_distance, 
                                        max_cell_circumradius=max_cell_circumradius, 
                                        verbose=False)

        tetmesh = cls()
        tetmesh._mesh = mesh
        tetmesh._set_fem_matrices()
        tetmesh._expand_mesh_data()

        return tetmesh

    @classmethod
    def generate_mesh_from_numpy_array(cls, array, voxel_size, max_cell_circumradius, max_facet_distance):
        """Generate a mesh from a numpy array using `the pygalmesh package`_:
        
        .. _the pygalmesh package: https://github.com/nschloe/pygalmesh
        Args:
            array (:obj:`numpy array`): Numpy array to generate mesh from.
            voxel_size (:obj:`float`): Dimension of array voxels.
            max_cell_circumradius (:obj:`float`): Bound for element radii.
            max_facet_distance (:obj:`float`): Bound for facet distance.
        """

        mesh = pygalmesh.generate_from_array( array, [voxel_size]*3,
                                            max_facet_distance=max_facet_distance, 
                                            max_cell_circumradius=max_cell_circumradius, 
                                            verbose=False )
        tetmesh = cls()
        tetmesh._mesh = mesh
        tetmesh._set_fem_matrices()
        tetmesh._expand_mesh_data()

        return tetmesh

    def _set_fem_matrices(self):
        """Extract and set mesh FEM matrices from pygalmesh object.
        """
        self.coord      = np.array(self._mesh.points)
        self.enod       = np.array(self._mesh.cells_dict['tetra'])
        self.dof        = np.arange(0,self.coord.shape[0]*3).reshape(self.coord.shape[0],3)
        self.nodal_coordinates = self.coord
        self.coefficents = np.zeros(self.coord.shape)

    def _expand_mesh_data(self):
        """Compute extended mesh quanteties such as element faces and normals.
        """
        self.efaces          = self._compute_mesh_faces( self.enod )
        self.enormals        = self._compute_mesh_normals( self.coord, self.enod, self.efaces )
        self.ecentroids      = self._compute_mesh_centroids( self.coord, self.enod ) 
        self.eradius         = self._compute_mesh_radius( self.coord, self.enod, self.ecentroids )
        self.ecmat           = self._compute_mesh_interpolation_matrices( self.enod, self.coord )

    def move( self, displacement ):
        """Update the mesh coordinates and any dependent quanteties by nodal displacements.
        Args:
            displacement (:obj:`numpy array`): Nodal displacements, ``shape=coefficents.shape``.
        """
        self._mesh.points      += displacement
        self.coord             = np.array(self._mesh.points)
        self.nodal_coordinates = self.coord
        self.enormals          = self._compute_mesh_normals( self.coord, self.enod, self.efaces )
        self.ecentroids        = self._compute_mesh_centroids( self.coord, self.enod ) 
        self.eradius           = self._compute_mesh_radius( self.coord, self.enod, self.ecentroids )
        self.ecmat             = self._compute_mesh_interpolation_matrices( self.enod, self.coord )

    def get_bounding_sphere_radius(self, node):
        """This method overrides :meth:`Basis.get_bounding_sphere_radius`.
        """
        elements = np.where( self.enod==node )[0]
        nodes = np.unique( self.enod[elements,:] )
        return np.max( np.linalg.norm( self.coord[nodes,:] - self.coord[node,:], axis=1 ) ) + 1e-8

    def _compute_mesh_interpolation_matrices( self, enod, coord ):
        """compute tetra element inverse C matrices.
        """
        interpolation_matrices = np.zeros((enod.shape[0],4,4))
        for element in range(enod.shape[0]):
            ec = coord[enod[element],:]
            V = np.ones( (ec.shape[0],ec.shape[0]) )
            V[1:,:] = ec.T[:,:]
            interpolation_matrices[element,:,:] = np.linalg.inv( V ).T
        return interpolation_matrices

    def _compute_interpolation_matrices( self, enod, coord ):
        """compute tetra element inverse C matrices.
        """
        interpolation_matrices = np.zeros((enod.shape[0],4,4))
        for element in range(enod.shape[0]):
            ec = coord[enod[element],:]
            V = np.ones( (ec.shape[0],ec.shape[0]) )
            V[1:,:] = ec.T[:,:]
            interpolation_matrices[element,:,:] = np.linalg.inv( V ).T
        return interpolation_matrices

    def _compute_mesh_faces(self, enod):
        """Compute all element faces nodal numbers.
        """
        efaces = np.zeros( (enod.shape[0], 4, 3), dtype=np.int )
        for i in range( enod.shape[0] ):
            # nodal combinations defining 4 unique planes in a tet.
            permutations = [ [0,1,2], [0,1,3], [0,2,3], [1,2,3] ]
            for j,perm in enumerate( permutations ):
                efaces[i,j,:] = enod[i,perm]
        return efaces

    def _compute_mesh_normals(self, coord, enod, efaces):
        """Compute all element faces outwards unit vector normals.
        """
        enormals = np.zeros( (enod.shape[0], 4, 3) )
        for i in range( enod.shape[0] ):
            ec = coord[enod[i,:], :]
            ecentroid = np.mean( ec, axis=0 )
            for j in range( efaces.shape[1] ): 
                ef = coord[efaces[i,j,:],:]
                enormals[i,j,:] = self._compute_plane_normal( ef, ecentroid )
        return enormals

    def _compute_plane_normal( self, points, centroid ):
        """Compute plane normal (outwards refering to centroid).
        """
        v1 = points[1,:] - points[0,:]
        v2 = points[2,:] - points[0,:]
        n  = np.cross(v1, v2)                            # define a vector perpendicular to the plane.
        n  = n*np.sign( n.dot(points[0,:] - centroid) )  # set vector direction outwards from centroid.   
        return n/np.linalg.norm(n)                       # normalise vector and return.

    def _compute_mesh_centroids(self, coord, enod):
        """Compute centroids of elements.
        """
        ecentroids = np.zeros((enod.shape[0],3))
        for i in range( enod.shape[0] ):
            ec = coord[enod[i,:], :]
            ecentroids[i,:] = np.sum( ec, axis=0 )/ec.shape[0]
        return ecentroids

    def _compute_mesh_radius(self, coord, enod, ecentroids):
        """Compute per element bounding radius.
        """
        eradius = np.zeros((enod.shape[0],))
        for i in range( enod.shape[0] ):
            ec = coord[enod[i,:], :]
            r  = np.linalg.norm( ec - ecentroids[i,:], axis=1)
            eradius[i] = np.max( r )
        return eradius

    def __call__(self, X, Y, Z, dim='all'):
        """This method overrides :meth:`Basis.__call__`.
        """
        shape = X.shape
        xs, ys, zs = X.flatten().astype(np.float64), Y.flatten().astype(np.float64), Z.flatten().astype(np.float64)

        if dim=='all':
            values = _get_interpolation_values_nd( xs, ys, zs, 
                                            self.ecentroids.astype(np.float64), 
                                            self.eradius.astype(np.float64), 
                                            self.enormals, 
                                            self.coord.astype(np.float64), 
                                            self.efaces, 
                                            self.ecmat, 
                                            self.enod, 
                                            self.coefficents.astype(np.float64) )
            return values.reshape(shape+(3,))
        else:
            values = _get_interpolation_values_1d( xs, ys, zs, 
                                            self.ecentroids.astype(np.float64), 
                                            self.eradius.astype(np.float64), 
                                            self.enormals, 
                                            self.coord.astype(np.float64), 
                                            self.efaces, 
                                            self.ecmat, 
                                            self.enod, 
                                            self.coefficents.astype(np.float64),
                                            dim )
            return values.reshape(shape)           

    def to_xdmf(self, file):
        """Save the tetra mesh to .xdmf paraview readable format for visualisation.
        Args:
            file (:obj:`str`): Absolute path to save the mesh at (without .xdmf extension)
        """
        self._mesh.write(file+".xdmf")