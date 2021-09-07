import numpy as np
import os, sys

def get_planar_rodriguez_rotator( v1, v2 ):
    """Return a function for rotating a vector v1 in the plane described by v1 and v2 towards v2 a fraction s=[0,1].
    """
    v1hat = v1/np.linalg.norm(v1)
    v2hat = v2/np.linalg.norm(v1)
    rhat  = np.cross(v1hat, v1hat)
    alpha = np.acos(v1hat.dot(v2hat))
    def rotator(v, s): return v*np.cos( s*alpha ) + np.cross( rhat, v )*np.sin( s*alpha )
    return rotator

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
