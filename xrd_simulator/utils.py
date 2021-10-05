import numpy as np
import os, sys

class RodriguezRotator(object):
    """Object for rotating vectors in the plane described by v1 and v2 towards v2 a fraction s=[0,1].
    
    Args:
        v1 (:obj:`numpy array`): A vector in 3d euclidean space (```shape=(3,)```)
        v2 (:obj:`numpy array`): A vector in 3d euclidean space (```shape=(3,)```)

    Attributes:
        rhat (:obj:`numpy array`): Unit vector orthogonal to both v1 and v2  (```shape=(3,)```)
        alpha (:obj:`float`): Angle in radians between v1 and v2.

    """ 

    def __init__(self, v1, v2):
        v1hat = v1/np.linalg.norm(v1)
        v2hat = v2/np.linalg.norm(v2)
        r  = np.cross(v1hat, v2hat)
        self.rhat = r / np.linalg.norm(r)
        self.alpha = np.arccos( v1hat.dot(v2hat) )
        rx,ry,rz = self.rhat
        self.K = np.array([ [ 0, -rz,  ry],
                            [ rz,  0, -rx],
                            [ -ry, rx,  0] ])
        self.K2 = self.K.dot( self.K )
        self.I  = np.eye(3,3)
        assert np.degrees( self.alpha ) > 1e-6, "Rotator has close to zero rotation intervall"
        assert np.degrees( self.alpha ) < 180,  "Rotator has close to 180dgrs rotation intervall"

    def get_rotation_matrix(self, s):
        return self.I + np.sin( s*self.alpha )*self.K + ( 1 - np.cos( s*self.alpha ) )*self.K2

    def __call__( self, vectors, s ):
        """Rotate a vector in the plane described by v1 and v2 towards v2 a fraction s=[0,1].
        
        Args:
            vectors (:obj:`numpy array`): A set of vectors in 3d euclidean space to be rotated (```shape=(3,N)```)
            s (:obj:`float`): Fraction to rotate, s=0 leaves the vector untouched while s=1 rotates it 
                alpha (:obj:`float`) radians.

        Returns:
            Rotated vector (:obj:`numpy array`) of ```shape=(3,)```.

        """ 
        R = self.get_rotation_matrix(s)
        return R.dot(vectors)

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

