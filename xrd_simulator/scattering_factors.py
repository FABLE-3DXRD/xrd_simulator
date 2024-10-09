import numpy as np
import pandas as pd
import torch
from xrd_simulator.cuda import frame
if frame != np:
    frame.array = frame.tensor
    frame.degrees = frame.rad2deg


def lorentz(beam,rigid_body_motion,K_out_xyz):
    """Compute the Lorentz intensity factor for all reflections."""

    k = beam.wave_vector
    kp = K_out_xyz
    rot_axis = frame.array(rigid_body_motion.rotation_axis,dtype=frame.float32)
    k_kp_norm = frame.matmul(k,kp.T) / (frame.linalg.norm(k,axis=0) * frame.linalg.norm(kp,axis=1))
    theta = frame.arccos(k_kp_norm) / 2.0
    korthogonal = kp - k_kp_norm.reshape(-1,1)*k.reshape(1,3)
    eta = frame.arccos(frame.matmul(rot_axis,korthogonal.T) / frame.linalg.norm(korthogonal))
    tol = 0.5
    condition = (frame.abs(frame.degrees(eta)) < tol) | (frame.degrees(eta) < tol) | (frame.abs(frame.degrees(eta)) > 180 - tol)
    infs = frame.where(condition, frame.inf, 0)
    return infs + 1.0 / (frame.sin(2 * theta) * frame.abs(frame.sin(eta)))

def polarization(beam,K_out_xyz):
    """Compute the Polarization intensity factor for all reflections."""

    kp_norm = K_out_xyz / frame.linalg.norm(K_out_xyz)
    return 1 - frame.matmul(beam.polarization_vector, kp_norm.T) ** 2
