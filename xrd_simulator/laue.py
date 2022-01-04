"""Collection of functions for solving the Laue equations for package specific parametrization.
"""

import numpy as np


def get_G(U, B, G_hkl):
    """Compute the diffraction vector G=UBG_HKL

    Args:
        U (:obj:`numpy array`) Orientation matrix of ``shape=(3,3)`` (unitary).
        B (:obj:`numpy array`): Reciprocal to grain coordinate mapping matrix of ``shape=(3,3)``.
        G_hkl (:obj:`numpy array`): Miller indices, i.e the h,k,l integers (``shape=(3,n)``).


    Returns:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector. (``shape=(3,n)``)

    """
    return np.dot(np.dot(U, B), G_hkl)


def get_bragg_angle(G, wavelength):
    """Compute a Bragg angle given a diffraction (scattering) vector.

    Args:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector. (``shape=(3,n)``)
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.

    Returns:
        Bragg angles (:obj:`float`): in units of radians. (``shape=(n,)``)

    """
    return np.arcsin(np.linalg.norm(G, axis=0) * wavelength / (4 * np.pi))


def get_sin_theta_and_norm_G(G, wavelength):
    """Compute a Bragg angle given a diffraction (scattering) vector.

    Args:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector.
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.

    Returns:
        sin(Bragg angle) (:obj:`float`): in units of radians and ||G||.

    """
    normG = np.linalg.norm(G, axis=0)
    return normG * wavelength / (4 * np.pi), normG


def get_tangens_half_angle_equation(k1, theta, G, rhat):
    """Find coefficient to the equation

    .. math::
        c_0 \\cos(s \\alpha) + c_1 \\sin(s \\alpha) + c_2 = 0. \\quad\\quad (1)

    """
    c_0 = np.dot(k1, G)
    c_1 = np.dot(np.cross(rhat, k1), G)
    c_2 = np.linalg.norm(k1) * np.linalg.norm(G) * np.sin(theta)
    return c_0, c_1, c_2


def find_solutions_to_tangens_half_angle_equation(c_0, c_1, c_2, alpha):
    """Find all solutions, :obj:`s`, to the equation (maximum 2 solutions exists)

    .. math::
        c_0 \\cos(s \\alpha) + c_1 \\sin(s \\alpha) + c_2 = 0. \\quad\\quad (1)

    by rewriting as

    .. math::
        (c_2 - c_0) t^2 + 2 c_1 t + (c_0 + c_2) = 0. \\quad\\quad (2)

    where

    .. math::
        t = \\tan(s \\alpha / 2). \\quad\\quad (3)

    and .. math::\\alpha is the angle between k1 and k2

    Args:
        c_0,c_1,c_2 (:obj:`float`): Coefficients c_0,c_1 and c_2 of equation (1).

    Returns:
        (:obj:`tuple` of :obj:`float` or :obj:`None`): solutions if existing otherwise returns None.

    """

    if c_0 == c_2:
        if c_1 == 0:
            t1 = t2 = None
        else:
            t1 = -c_0 / c_1
            t2 = None
    else:
        rootval = (c_1 / (c_2 - c_0))**2 - (c_0 + c_2) / (c_2 - c_0)
        leadingterm = (-c_1 / (c_2 - c_0))
        if rootval < 0:
            t1, t2 = None, None
        else:
            t1 = leadingterm + np.sqrt(rootval)
            t2 = leadingterm - np.sqrt(rootval)

    s1, s2 = None, None

    if t1 is not None:
        s1 = 2 * np.arctan(t1) / alpha
        if s1 > 1 or s1 < 0:
            s1 = None

    if t2 is not None:
        s2 = 2 * np.arctan(t2) / alpha
        if s2 > 1 or s2 < 0:
            s2 = None

    return s1, s2
