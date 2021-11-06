import numpy as np
import os, sys

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

def clip_line_with_convex_polyhedron( ray_points, ray_direction, plane_points, plane_normals ):
    """Compute clip-lengths of line-segments clipped by a convex polyhedron defined by a series of 2d planes.

        For algorihtm description see:
            Mike Cyrus and Jay Beck. “Generalized two- and three-dimensional clipping”. (1978)

        Args:
            ray_points (:obj:`numpy array`): base points of rays (exterior to polyhedron), ```shape=(n,3)```
            ray_direction  (:obj:`numpy array`): normalised ray directions,  ```shape=(n,3)```
            plane_points (:obj:`numpy array`): point on each polyhedron face of, ```shape=(m,3)```
            plane_normals (:obj:`numpy array`): outwards element face normals, shape: nbr elements x nbr faces x 3

        Returns:
            clip_lengths (:obj:`numpy array`) : intersection lengths.

    """
    clip_lengths = np.zeros((ray_points.shape[0],))

    # for each line segment
    for i,e in enumerate( ray_points ): 

        # find paramteric line-plane intersect based on orthogonal equations: (p - e - t*r) . n = 0 
        t1 = np.sum( np.multiply( plane_points-e, plane_normals ), axis=1 )
        t2 = np.dot( plane_normals, ray_direction ) 
        ti = t1/t2
        
        # Sort intersections as potential entry and exit points
        te = np.max( ti[t2<0] )
        tl = np.min( ti[t2>0] )

        clip_lengths[i] = np.max( [0, tl-te] )

    return clip_lengths
