import numpy as np
import torch
torch.set_default_dtype(torch.float64)

def lorentz(k_in, k_out, rot_axis):
    """Compute the Lorentz intensity factor for all reflections.
    
    Args:
        k_in: torch.Tensor of shape (3,)
        k_out: torch.Tensor of shape (N, 3) or (3,)
        rotation_axis: torch.Tensor of shape (3,)
    """
    # Handle both single vector and batch inputs efficiently
    kp = k_out.reshape(-1, 3) if k_out.dim() == 1 else k_out
    
    # Compute theta using normalized vectors
    k_in_norm = k_in / torch.linalg.norm(k_in)
    cos_theta = torch.matmul(k_in_norm, kp.T)
    theta = torch.arccos(cos_theta) / 2.0
    
    # Compute korthogonal
    korthogonal = kp - torch.matmul(kp, k_in_norm.reshape(3, 1)) * k_in_norm
    korth_norm = torch.linalg.norm(korthogonal, dim=1)
    
    # Use normalized vectors for eta calculation
    cos_eta = torch.matmul(rot_axis, korthogonal.T) / korth_norm
    eta = torch.arccos(cos_eta.clamp(-1, 1))  # Prevent numerical instabilities
    
    # Compute result with tolerance check
    tol = 0.5
    condition = ((torch.abs(torch.rad2deg(eta)) < tol) | 
                (torch.abs(torch.rad2deg(eta)) > 180 - tol) |
                (torch.rad2deg(theta) < tol))
    
    lorentz = 1.0 / (torch.sin(2 * theta) * torch.abs(torch.sin(eta)))
    return torch.where(condition, torch.tensor(float('inf')), lorentz).squeeze()


def polarization(beam,K_out_xyz):
    """Compute the Polarization intensity factor for all reflections."""

    kp_norm = K_out_xyz / torch.linalg.norm(K_out_xyz)
    return 1 - torch.matmul(beam.polarization_vector, kp_norm.T) ** 2
