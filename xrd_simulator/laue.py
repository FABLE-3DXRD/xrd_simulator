"""Collection of functions for solving time-dependent Laue equations.

This module provides functions for arbitrary rigid body motions. It is mainly
used internally by the :class:`xrd_simulator.polycrystal.Polycrystal`. However,
for the advanced user, access to these functions may be of interest.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
from xrd_simulator.utils import ensure_torch

def _get_G(U, B, G_hkl):
    """Compute the diffraction vector.

    .. math::
        \boldsymbol{G} = \boldsymbol{U}\boldsymbol{B}\boldsymbol{G}_{hkl}

    Parameters
    ----------
    U : numpy.ndarray
        Orientation matrix of shape ``(3, 3)`` (unitary).
    B : numpy.ndarray
        Reciprocal to grain coordinate mapping matrix of shape ``(3, 3)``.
    G_hkl : numpy.ndarray
        Miller indices, i.e. the h, k, l integers, shape ``(3, n)``.

    Returns
    -------
    numpy.ndarray
        Sample coordinate system diffraction vector, shape ``(3, n)``.
    """

    U = ensure_torch(U)
    B = ensure_torch(B)
    G_hkl = ensure_torch(G_hkl)
    
    # Handle 1D input (single plane) by converting to column vector
    if G_hkl.dim() == 1:
        G_hkl = G_hkl.unsqueeze(0)  # (3,) -> (1, 3)
    
    # G_hkl is (n, 3), transpose to (3, n) for matmul
    return torch.matmul(torch.matmul(U, B), G_hkl.mT)



def _find_solutions_to_tangens_half_angle_equation(
    G_0, rho_0_factor, rho_1_factor, rho_2_factor, delta_omega):
    """Find all solutions, t, to the tangent half-angle equation.

    Solves the equation (maximum 2 solutions exist)::

        rho_0 * cos(t * delta_omega) + rho_1 * sin(t * delta_omega) + rho_2 = 0     (1)

    by rewriting it as a quadratic equation in terms of s::

        (rho_2 - rho_0) * s^2 + 2 * rho_1 * s + (rho_0 + rho_2) = 0                 (2)

    where ``s = tan(t * delta_omega / 2)``.

    Parameters
    ----------
    G_0 : numpy.ndarray
        The non-rotated scattering vectors for all tetrahedra of a given phase.
        Dimensions should be ``(tetrahedra, coordinates, hkl_planes)``.
    rho_0_factor : numpy.ndarray
        Factors to compute rho_0 of equation (1).
    rho_1_factor : numpy.ndarray
        Factors to compute rho_1 of equation (1).
    rho_2_factor : numpy.ndarray
        Factors to compute rho_2 of equation (1).
    delta_omega : float
        Radians of rotation.

    Returns
    -------
    tuple
        A tuple ``(grains, planes, times, G)`` containing:

        - ``grains``: Grain indices for each solution.
        - ``planes``: Plane indices for each solution.
        - ``times``: Diffraction times for each solution.
        - ``G``: Diffraction vectors for each solution.
    """

    # Convert inputs to tensors
    G_0 = torch.asarray(G_0, dtype=torch.float64)
    rho_0_factor = torch.asarray(rho_0_factor, dtype=torch.float64)
    rho_1_factor = torch.asarray(rho_1_factor, dtype=torch.float64)
    rho_2_factor = torch.asarray(rho_2_factor, dtype=torch.float64)

    # Ensure G_0 has at least 3 dimensions
    if len(G_0.shape) == 2:
        G_0 = G_0[torch.newaxis, :, :]

    # Compute rho_0 and rho_2
    rho_0 = torch.matmul(rho_0_factor, G_0)
    rho_2 = torch.matmul(rho_2_factor, G_0) + torch.sum(G_0**2, axis=1) / 2.0
    denominator = rho_2 - rho_0
    numerator = rho_2 + rho_0

    del rho_2
    #Remove 0 denominators
    denominator[denominator==0] = torch.nan

    # Calculate coefficients for quadratic equation
    a = torch.divide(
        torch.matmul(rho_1_factor, G_0),
        denominator,
        out=torch.full_like(rho_0, torch.nan)
    )

    b = torch.divide(
        numerator, denominator, out=torch.full_like(rho_0, torch.nan)
    )

    # Clean up unnecessary variables

    del denominator, numerator, rho_0

    # Calculate discriminant

    discriminant = a**2 - b
    del b

    # Handle cases where discriminant is negative
    discriminant[discriminant<0] = torch.nan
    # discriminant[discriminant>10] = torch.nan 

    # Calculate solutions for s
    s1 = -a + torch.sqrt(discriminant)
    s2 = -a - torch.sqrt(discriminant)
    del a, discriminant
    t1 = 2 * torch.arctan(s1) / delta_omega
    del s1
    indices_t1 = torch.argwhere(torch.logical_and(t1 >= 0, t1 <= 1))
    values_t1 = t1[indices_t1[:,0], indices_t1[:,1]]
    del t1
    t2 = 2 * torch.arctan(s2) / delta_omega
    del s2, delta_omega
    indices_t2 = torch.argwhere(torch.logical_and(t2 >= 0, t2 <= 1))
    values_t2 = t2[indices_t2[:,0], indices_t2[:,1]]
    del t2
    peak_index = torch.concatenate((indices_t1, indices_t2), axis=0)
    del indices_t1,indices_t2
    times = torch.concatenate((values_t1, values_t2), axis=0)
    del values_t1,values_t2
    grains = peak_index[:, 0]
    planes = peak_index[:, 1]
    del peak_index
    G_0 = torch.transpose(G_0,2,1)    
    G = G_0[grains, planes]

    return grains, planes, times, G


# ==============================================================================
# DEPRECATED METHODS - TO BE REMOVED IN FUTURE VERSION
# ==============================================================================
# The following methods are no longer called anywhere in the codebase.
# They are kept temporarily for backwards compatibility but will be removed.
# ==============================================================================

def _get_bragg_angle(G, wavelength):
    """Compute a Bragg angle given a diffraction (scattering) vector.

    .. deprecated::
        This method is no longer used in the codebase and will be removed
        in a future version.

    Parameters
    ----------
    G : numpy.ndarray
        Sample coordinate system diffraction vector, shape ``(3, n)``.
    wavelength : float
        Photon wavelength in units of angstrom.

    Returns
    -------
    numpy.ndarray
        Bragg angles in units of radians, shape ``(n,)``.
    """
    G = ensure_torch(G)
    return torch.arcsin(torch.linalg.norm(G, axis=0) * wavelength / (4 * np.pi))


def _get_sin_theta_and_norm_G(G, wavelength):
    """Compute sin(Bragg angle) and norm of the diffraction vector.

    .. deprecated::
        This method is no longer used in the codebase and will be removed
        in a future version.

    Parameters
    ----------
    G : numpy.ndarray
        Sample coordinate system diffraction vector.
    wavelength : float
        Photon wavelength in units of angstrom.

    Returns
    -------
    tuple
        ``(sin_theta, norm_G)`` where:

        - ``sin_theta``: Sine of the Bragg angle.
        - ``norm_G``: Norm of the diffraction vector.
    """
    G = ensure_torch(G)
    normG = torch.linalg.norm(G, axis=0)
    return normG * wavelength / (4 * np.pi), normG
