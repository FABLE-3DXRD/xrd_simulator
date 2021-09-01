import numpy as np

def get_vector_seperation_angle(v1, v2):
    """Compute angle in radians between vector v1 and v2.
    """
    return np.acos(  v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) )

def planar_rodriguez_rotation( v1, v2, s ):
    """Rotate vector v1 in the plane described by v1 and v2 towards v2 a fraction s=[0,1]
    """
    r = np.cross(v1, v2)
    r /= np.linalg.norm(r)
    alpha = get_vector_seperation_angle(v1, v2)
    return v1*np.cos( s*alpha ) + np.cross( r, v1 )*np.sin( s*alpha )