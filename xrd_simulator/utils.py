import numpy as np
import os, sys

def get_unit_vector_and_l2norm(v1, v2):
    """Compute l2 norm distance and unit vector between vectors v2 and v1.
    """
    v2v1 = (v2 - v1)
    norm = np.linalg.norm( v2v1 ) 
    unit_vector = v2v1 / norm
    return unit_vector, norm

class PlanarRodriguezRotator(object):

    """For rotating a vector v1 in the plane described by v1 and v2 towards v2 a fraction s=[0,1].
    """

    def __init__(self, v1, v2):
        v1hat = v1/np.linalg.norm(v1)
        v2hat = v2/np.linalg.norm(v2)
        r  = np.cross(v1hat, v2hat)
        self.rhat = r / np.linalg.norm(r)
        self.alpha = np.arccos( v1hat.dot(v2hat) )
        assert np.degrees( self.alpha ) > 1e-6, "Rotator has close to zero rotation intervall"
        assert np.degrees( self.alpha ) < 180,  "Rotator has close to 180dgrs rotation intervall"

    def __call__( self, v, s ):
        return v*np.cos( s*self.alpha ) + np.cross( self.rhat, v )*np.sin( s*self.alpha )

class HiddenPrints:
    """Simple class to enable running code without printing using python with statements.

    This is a hack to supress printing from imported packages over which we do not have full controll. (xfab.tools)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
