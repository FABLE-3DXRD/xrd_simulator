import numpy as np
import torch

def lorentz(beam,rigid_body_motion,K_out_xyz):
    """Compute the Lorentz intensity factor for all reflections."""

    k = beam.wave_vector
    kp = K_out_xyz
    rot_axis = rigid_body_motion.rotation_axis
    k_kp_norm = torch.matmul(k,kp.T) / (torch.linalg.norm(k,axis=0) * torch.linalg.norm(kp,axis=1))
    theta = torch.arccos(k_kp_norm) / 2.0
    korthogonal = kp - k_kp_norm.reshape(-1,1)*k.reshape(1,3)
    eta = torch.arccos(torch.matmul(rot_axis,korthogonal.T) / torch.linalg.norm(korthogonal))
    tol = 0.5
    #condition = (torch.abs(torch.degrees(eta)) < tol) | (torch.degrees(eta) < tol) | (torch.abs(torch.degrees(eta)) > 180 - tol)
    #infs = torch.where(condition, torch.inf, 0)
    return 1.0 / (torch.sin(2 * theta) * torch.abs(torch.sin(eta)))

def polarization(beam,K_out_xyz):
    """Compute the Polarization intensity factor for all reflections."""

    kp_norm = K_out_xyz / torch.linalg.norm(K_out_xyz)
    return 1 - torch.matmul(beam.polarization_vector, kp_norm.T) ** 2
