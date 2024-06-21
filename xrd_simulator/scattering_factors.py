import numpy as np
import pandas as pd
import torch

def lorentz(beam,rigid_body_motion,peaks_df):
    """Compute the Lorentz intensity factor for a scattering_unit."""

    k = beam.wave_vector
    kp = peaks_df[["k'x", "k'y", "k'z"]]
    k_kp_norm = k.dot(kp.T) / (np.linalg.norm(k,axis=0) * np.linalg.norm(kp,axis=1))
    theta = np.arccos(k_kp_norm) / 2.0
    korthogonal = kp - k_kp_norm.reshape(-1,1)*k.reshape(1,3)
    eta = np.arccos(rigid_body_motion.rotation_axis.dot(korthogonal.T) / np.linalg.norm(korthogonal))
    tol = 0.5
    condition = np.array((np.abs(np.degrees(eta)) < tol) | (np.degrees(eta) < tol) | (np.abs(np.degrees(eta)) > 180 - tol))
    infs = np.where(condition, np.inf, 0)
    return infs + 1.0 / (np.sin(2 * theta) * np.abs(np.sin(eta)))
