import numpy as np
import pandas as pd
import torch
from xrd_simulator.cuda import fw
if fw != np:
    fw.array = fw.tensor
    fw.degrees = fw.rad2deg


def lorentz(beam,rigid_body_motion,K_out_xyz):
    """Compute the Lorentz intensity factor for all reflections."""

    k = beam.wave_vector
    kp = K_out_xyz
    rot_axis = fw.array(rigid_body_motion.rotation_axis,dtype=fw.float32)
    k_kp_norm = fw.matmul(k,kp.T) / (fw.linalg.norm(k,axis=0) * fw.linalg.norm(kp,axis=1))
    theta = fw.arccos(k_kp_norm) / 2.0
    korthogonal = kp - k_kp_norm.reshape(-1,1)*k.reshape(1,3)
    eta = fw.arccos(fw.matmul(rot_axis,korthogonal.T) / fw.linalg.norm(korthogonal))
    tol = 0.5
    condition = (fw.abs(fw.degrees(eta)) < tol) | (fw.degrees(eta) < tol) | (fw.abs(fw.degrees(eta)) > 180 - tol)
    infs = fw.where(condition, fw.inf, 0)
    return infs + 1.0 / (fw.sin(2 * theta) * fw.abs(fw.sin(eta)))

def polarization(beam,K_out_xyz):
    """Compute the Polarization intensity factor for all reflections."""

    kp_norm = K_out_xyz / fw.linalg.norm(K_out_xyz)
    return 1 - fw.matmul(beam.polarization_vector, kp_norm.T) ** 2
