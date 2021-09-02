import numpy as np

def get_planar_rodriguez_rotator( v1, v2 ):
    """Return a function for rotating a vector v1 in the plane described by v1 and v2 towards v2 a fraction s=[0,1].
    """
    v1hat = v1/np.linalg.norm(v1)
    v2hat = v2/np.linalg.norm(v1)
    rhat  = np.cross(v1hat, v1hat)
    alpha = np.acos(v1hat.dot(v2hat))
    def rotator(v, s): return v*np.cos( s*alpha ) + np.cross( rhat, v )*np.sin( s*alpha )
    return rotator