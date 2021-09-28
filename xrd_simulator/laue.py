"""Collection of functions for solving the Laue equations for package specific parametrisations.
"""

import numpy as np

def get_G( U, B, G_hkl ):
    """Compute the diffraction vector G=UBG_HKL

    Args:
        U (:obj:`numpy array`) Orientation matrix of ```shape=(3,3)``` (unitary).
        B (:obj:`numpy array`): Reciprocal to grain coordinate mapping matrix of ```shape=(3,3)```.
        G_hkl (:obj:`numpy array`): Miller indices, i.e the h,k,l integers (```shape=(3,)```).


    Returns:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector.

    """
    return np.dot( U, np.dot( B, G_hkl ) )

def get_bragg_angle( G, wavelength ):
    """Compute a Bragg angle given a diffraction (scattering) vector.

    Args:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector.
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.

    Returns:
        Bragg angle (:obj:`float`): in units of radians.
        
    """
    return np.arcsin( np.linalg.norm(G)*wavelength/(4*np.pi) )

def get_tangens_half_angle_equation( k1, theta, G, rhat ):
    """Find the coefficents of the quadratic equation 

    .. math::
        (c_2 - c_0) t^2 + 2 c_1 t + (c_0 + c_2) = 0. \\quad\\quad (1)

    Its roots (if existing) are solutions to the Laue equations where t is

    .. math::
        t = \\tan(s \\alpha / 2). \\quad\\quad (2)

    and s is the sought parameter which parametrises all wave vectors between k1 and k2 where 
    s=0 corresponds to k1 and s=1 to k2. Intermediate vectors are reached by rotating k1 in the
    plane spanned by k1 and k2.

    Args:
        k1 (:obj:`numpy array`): Beam wavevector for s=0 with ```shape=(3,)```
        theta (:obj:`float`): Bragg angle in radians.
        G (:obj:`numpy array`): Sample coordinate system diffraction vector.
        rhat (:obj:`numpy array`): Unit vector orthogonal to both k1 and k2.

    Returns:
        (:obj:`tuple` of :obj:`float`): Coefficents c_0,c_1 and c_2 of equation (1).

    """
    # TODO: Optimize this function, np.cross( rhat, k1 ) is slow and should be precomputed and passed
    # in as it is not a function of th scattering but of the beam selection. Also make this function
    # support arrays of G and theta.
    c_0  = np.dot(G, k1)
    c_1  = np.dot(G, np.cross( rhat, k1 ) )
    c_2  = np.linalg.norm( k1 ) * np.linalg.norm( G ) * np.sin( theta )
    return (c_0, c_1, c_2)

def find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha ):
    """Find all solutions, :obj:`s`, to the equation (maximum 2 solutions exists)

    .. math::
        (c_2 - c_0) t^2 + 2 c_1 t + (c_0 + c_2) = 0. \\quad\\quad (1)

    where

    .. math::
        t = \\tan(s \\alpha / 2). \\quad\\quad (2)
    
    and .. math::\\alpha is the angle between k1 and k2

    Args:
        c_0,c_1,c_2 (:obj:`float`): Coefficents c_0,c_1 and c_2 of equation (1).

    Returns:
        (:obj:`tuple` of :obj:`float` or :obj:`None`): solutions if any exists otherwise returns None.

    """

    if c_0==c_2:
        if c_1==0:
            t1 = t2 = None
        else:
            t1 = -c_0/c_1
    else:
        rootval = (c_1/(c_2 - c_0))**2 -  (c_0 + c_2)/(c_2 - c_0)
        leadingterm = ( -c_1/(c_2 - c_0) ) 
        if rootval<0:
            t1,t2 = None, None
        else:
            t1 = leadingterm + np.sqrt( rootval )
            t2 = leadingterm - np.sqrt( rootval )

    s1, s2 = None, None

    if t1 is not None:
        s1 = 2 * np.arctan( t1 ) / alpha
        if s1>1 or s1<0:
            s1 = None

    if t2 is not None:
        s2 = 2 * np.arctan( t2 ) / alpha
        if s2>1 or s2<0:
            s2 = None


    return s1, s2