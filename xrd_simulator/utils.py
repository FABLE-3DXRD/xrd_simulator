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
