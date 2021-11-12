import numpy as np
import os, sys
from numba import njit
from scipy import optimize

class _HiddenPrints:
    """Simple class to enable running code without printing using python with statements.

    This is a hack to supress printing from imported packages over which we do not have full controll. (xfab.tools)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_unit_vector_and_l2norm(point_1, point_2):
    """Compute l2 norm distance and unit vector between vectors v2 and v1.

    Args:
        point_1 (:obj:`numpy array`): A point in 3d euclidean space (```shape=(3,)```)
        point_2 (:obj:`numpy array`): A point in 3d euclidean space (```shape=(3,)```)

    Returns:
        (:obj:`tuple` of :obj:`numpy array` and :obj:`float`) Unit vector  of ```shape=(3,)```
            from point_1 to point_2 and the distance between point_1 and point_2

    """ 
    p2p1 = (point_2 - point_1)
    norm = np.linalg.norm( p2p1 ) 
    unit_vector = p2p1 / norm
    return unit_vector, norm

def contained_by_intervals(s, intervals):
    for bracket in intervals:
        if s >= bracket[0] and s <= bracket[1]:
            return True
    return False

def CIFopen( ciffile ):
    from CifFile import ReadCif
    cf = ReadCif(ciffile)
    return cf[ list(cf.keys())[0] ]



@njit
def clip_line_with_convex_polyhedron( line_points, line_direction, plane_points, plane_normals ):
    """Compute clip-lengths of a set of parallel lines clipped by a convex polyhedron defined by a series of 2d planes.

        For algorihtm description see:
            Mike Cyrus and Jay Beck. “Generalized two- and three-dimensional clipping”. (1978)

        Args:
            line_points (:obj:`numpy array`): base points of rays (exterior to polyhedron), ```shape=(n,3)```
            line_direction  (:obj:`numpy array`): normalised ray direction (all rays have the same direction),  ```shape=(3,)```
            plane_points (:obj:`numpy array`): point in each polyhedron face plane, ```shape=(m,3)```
            plane_normals (:obj:`numpy array`): outwards element face normals, shape: ```shape=(m,3)```

        Returns:
            clip_lengths (:obj:`numpy array`) : intersection lengths.  ```shape=(n,)```

    """
    #TODO: Add test
    clip_lengths = np.zeros((line_points.shape[0],))
    t2 = np.dot( plane_normals, line_direction ) 
    te_mask = t2<0
    tl_mask = t2>0
    for i,line_point in enumerate( line_points ): 

        # find paramteric line-plane intersect based on orthogonal equations
        t1 = np.sum( np.multiply( plane_points-line_point, plane_normals ), axis=1 ) # 
        ti = t1/t2
        
        # Sort intersections as potential entry and exit points
        te = np.max( ti[te_mask] )
        tl = np.min( ti[tl_mask] )

        if tl>te:
            clip_lengths[i] = tl-te

    return clip_lengths
