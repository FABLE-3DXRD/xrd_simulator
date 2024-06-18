"""Collection of functions for solving time dependent Laue equations for arbitrary rigid body motions.
This module is mainly used internally by the :class:`xrd_simulator.polycrystal.Polycrystal`. However,
for the advanced user, access to these functions may be of interest.
"""
import numpy as np
import cupy as cp
import torch
from xrd_simulator import utils
from xrd_simulator.cuda import use_cuda

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

    if use_cuda:
        frame = torch
        U = frame.asarray(U,dtype=frame.float32)
        B = frame.asarray(B,dtype=frame.float32)
        G_hkl = frame.asarray(G_hkl,dtype=frame.float32)
    else:
        frame = np     
        U = U.astype(np.float32)
        B = B.astype(np.float32)
        G_hkl = G_hkl.astype(np.float32)
    return frame.matmul(frame.matmul(U, B), G_hkl.T)



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
    G_0, rho_0_factor, rho_1_factor, rho_2_factor, delta_omega):
    """
    Find all solutions, t, to the equation (maximum 2 solutions exist):

        rho_0 * cos(t * delta_omega) + rho_1 * sin(t * delta_omega) + rho_2 = 0.     (1)

    by rewriting it as a quadratic equation in terms of s:

        (rho_2 - rho_0) * s^2 + 2 * rho_1 * s + (rho_0 + rho_2) = 0.                  (2)

    where s = tan(t * delta_omega / 2).                                             (3)

    Args:
        G_0 (:obj:`numpy.ndarray`): The non-rotated scattering vectors for all tetrahedra of a given phase.
            Dimensions should be (tetrahedra, coordinates, hkl_planes).
        rho_0_factor (:obj:`numpy.ndarray` of :obj:`float`): Factors to compute rho_0 of equation (1).
        rho_1_factor (:obj:`numpy.ndarray` of :obj:`float`): Factors to compute rho_1 of equation (1).
        rho_2_factor (:obj:`numpy.ndarray` of :obj:`float`): Factors to compute rho_2 of equation (1).
        delta_omega (:obj:`float`): Radians of rotation.

    Returns:
        (:obj:`tuple` of :obj:`numpy.ndarray`): A tuple containing two numpy arrays:
            - indices: 2D numpy array representing indices for diffraction computation.
            - values: 1D numpy array representing values for diffraction computation.
    """

    # Transfer input arrays to GPU if they are not already there

    if use_cuda:
        frame=torch
        G_0 = frame.asarray(G_0)
        rho_0_factor = frame.asarray(rho_0_factor)
        rho_1_factor = frame.asarray(rho_1_factor)
        rho_2_factor = frame.asarray(rho_2_factor)
    else:
        frame=np     
        G_0 = cp.asnumpy(G_0)

    # Ensure G_0 has at least 3 dimensions
    if len(G_0.shape) == 2:
        G_0 = G_0[frame.newaxis, :, :]

    # Compute rho_0 and rho_2
    rho_0 = frame.matmul(rho_0_factor, G_0)
    rho_2 = frame.matmul(rho_2_factor, G_0) + frame.sum(G_0**2, axis=1) / 2.0
    # del rho_0_factor,rho_2_factor
    # Calculate constants for quadratic equation
    denominator = rho_2 - rho_0
    numerator = rho_2 + rho_0
    # del  rho_2

    # Calculate coefficients for quadratic equation
    a = frame.divide(
        frame.matmul(rho_1_factor, G_0),
        denominator,
        out=frame.full_like(rho_0, np.nan),
        #where= denominator != 0,
    )
    # del rho_1_factor
    b = frame.divide(
        numerator, denominator, out=frame.full_like(rho_0, np.nan)
    )

    # Clean up unnecessary variables
    # del denominator, numerator, rho_0

    # Calculate discriminant
    discriminant = a**2 - b
    # del b

    # Handle cases where discriminant is negative
    discriminant[discriminant < 0] = np.nan
    discriminant[frame.isinf(discriminant)] = np.nan

    # Calculate solutions for s
    s1 = -a + frame.sqrt(discriminant)
    s2 = -a - frame.sqrt(discriminant)

    # Clean up discriminant and a
    # del discriminant, a
    s = frame.concatenate((s1,s2),axis=0)
    # del s1,s2
    # Calculate solutions for t1 and t2
    t = 2 * frame.arctan(s) / delta_omega
    # del s,delta_omega

    # Filter solutions within range [0, 1]
    valid_t_indices = frame.logical_and(t >= 0, t <= 1)

    values_t = t[valid_t_indices]
    # del t
    reflection_index = frame.argwhere(valid_t_indices)
    reflection_index = reflection_index % G_0.shape[0]
    # del valid_t_indices
    # Return concatenated indices and values

    G_0 = frame.transpose(G_0,2,1)
    G_0_reflected = G_0[reflection_index[:, 0], reflection_index[:, 1]]

    # del G_0
    output = frame.concatenate((reflection_index, values_t.unsqueeze(1), G_0_reflected), axis=1)
    # del reflection_index,values_t,G_0_reflected

    output = output.cpu().numpy() if use_cuda else output
        
    return output
