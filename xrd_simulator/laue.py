"""Collection of functions for solving time dependent Laue equations for arbitrary rigid body motions.
This module is mainly used internally by the :class:`xrd_simulator.polycrystal.Polycrystal`. However,
for the advanced user, access to these functions may be of interest.
"""

import numpy as np


def get_G(U, B, G_hkl):
    """Compute the diffraction vector

    .. math::
        \\boldsymbol{G} = \\boldsymbol{U}\\boldsymbol{B}\\boldsymbol{G}_{hkl}

    Args:
        U (:obj:`numpy array`) Orientation matrix of ``shape=(3,3)`` (unitary).
        B (:obj:`numpy array`): Reciprocal to grain coordinate mapping matrix of ``shape=(3,3)``.
        G_hkl (:obj:`numpy array`): Miller indices, i.e the h,k,l integers (``shape=(3,n)``).


    Returns:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector. (``shape=(3,n)``)

    """

    return np.float32(np.matmul(np.matmul(U, B), G_hkl.T))


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
        norm_G (:obj:`float`): Norm of the diffraction vector.

    """
    normG = np.linalg.norm(G, axis=0)
    return normG * wavelength / (4 * np.pi), normG


def find_solutions_to_tangens_half_angle_equation(
    G_0, rho_0_factor, rho_1_factor, rho_2_factor, delta_omega
):
    """Find all solutions, :obj:`t`, to the equation (maximum 2 solutions exists)

        .. math::
            \\rho_0 \\cos(t \\Delta \\omega) + \\rho_1 \\sin(t \\Delta \\omega) + \\rho_2 = 0. \\quad\\quad (1)

        by rewriting as

        .. math::
            (\\rho_2 - \\rho_0) s^2 + 2 \\rho_1 s + (\\rho_0 + \\rho_2) = 0. \\quad\\quad (2)

        where

        .. math::
            s = \\tan(t \\Delta \\omega / 2). \\quad\\quad (3)
    #Computed in advance to be
            G_0: The non-rotated scattering vectors for all tetrahedra of a given phase. dimensions --> (tetrahedra,coordinates,hkl_planes)
            \\rho_0_factor,\\rho_1_factor,\\rho_2_factor (:obj:`float`): Factors to compute the \\rho_0,\\rho_1 and \\rho_2 of equation (1).
            delta_omega (:obj:`float`): Radians of rotation.

        Returns:
            (:obj:`tuple` of :obj:`numpy.array`): A tuple containing two numpy arrays:
            - indices: 2D numpy array representing indices for diffraction computation.
            - values: 1D numpy array representing values for diffraction computation.

    """

    if (
        len(G_0.shape) == 2
    ):  # We add an empty dimension first in case it's a single tet G_0 being passed.
        G_0 = G_0[np.newaxis, :, :]

    rho_0 = np.matmul(rho_0_factor, G_0)
    rho_2 = np.matmul(rho_2_factor, G_0) + np.sum((G_0 * G_0), axis=1) / 2.0
    denominator = rho_2 - rho_0
    numerator = rho_2 + rho_0
    del rho_2
    a = np.divide(
        np.matmul(rho_1_factor, G_0),
        denominator,
        out=np.full_like(rho_0, np.nan),
        where=denominator != 0,
    )
    b = np.divide(
        numerator, denominator, out=np.full_like(rho_0, np.nan), where=denominator != 0
    )
    del denominator, numerator, rho_0
    rootval = a**2 - b
    leadingterm = -a
    del a, b
    rootval[rootval < 0] = np.nan
    s1 = leadingterm + np.sqrt(rootval)
    s2 = leadingterm - np.sqrt(rootval)
    del rootval, leadingterm
    t1 = 2 * np.arctan(s1) / delta_omega
    del s1
    indices_t1 = np.array(np.where(np.logical_and(t1 >= 0, t1 <= 1)))
    values_t1 = t1[indices_t1[0, :], indices_t1[1, :]]

    del t1
    t2 = 2 * np.arctan(s2) / delta_omega
    del s2, delta_omega
    indices_t2 = np.array(np.where(np.logical_and(t2 >= 0, t2 <= 1)))
    values_t2 = t2[indices_t2[0, :], indices_t2[1, :]]
    del t2
    return np.concatenate((indices_t1, indices_t2), axis=1), np.concatenate(
        (values_t1, values_t2), axis=0
    )
