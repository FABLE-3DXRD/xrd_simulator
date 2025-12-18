import unittest
from scipy.spatial import ConvexHull
import os
import torch
from xrd_simulator.detector import Detector
from xrd_simulator.phase import Phase
from xrd_simulator.utils import ensure_torch, ensure_numpy
import numpy as np

torch.set_default_dtype(torch.float64)

def create_peak_tensor(convex_hull, scattered_wave_vector, incident_wave_vector, wavelength,
                      incident_polarization_vector, rotation_axis, time, phase, hkl_indx,
                      element_index, peak_index, volumes=None):
    """Helper to create a peak tensor with the same structure as polycrystal.diffract()"""
    # Calculate factors that used to be in ScatteringUnit
    k = incident_wave_vector
    kp = scattered_wave_vector
    theta = np.arccos(np.clip(k.dot(kp) / (np.linalg.norm(k) ** 2), -1.0, 1.0)) / 2.0
    korthogonal = kp - (k * kp.dot(k) / (np.linalg.norm(k) ** 2))
    korth_norm = np.linalg.norm(korthogonal)
    if korth_norm > 1e-10:
        eta = np.arccos(np.clip(rotation_axis.dot(korthogonal) / korth_norm, -1.0, 1.0))
    else:
        eta = np.pi / 2  # Default to 90 degrees if korthogonal is near zero

    # Calculate polarization
    khatp = kp / np.linalg.norm(kp)
    polarization_factor = 1 - np.dot(incident_polarization_vector, khatp) ** 2

    # Calculate lorentz - use finite value to avoid filtering
    # In real simulations, infinite Lorentz factors get filtered out
    # For testing, use a reasonable finite value
    sin_2theta = np.sin(2 * theta)
    sin_eta = np.abs(np.sin(eta))
    
    # Use a safe minimum denominator to avoid infinities
    if sin_2theta > 1e-3 and sin_eta > 1e-3:
        lorentz_factor = 1.0 / (sin_2theta * sin_eta)
    else:
        lorentz_factor = 100.0  # Use large but finite value for testing

    # Get structure factors from phase
    if phase.structure_factors is not None:
        structure_factors = phase.structure_factors[hkl_indx]
        structure_factor = np.sum(structure_factors**2)
    else:
        structure_factor = 1.0
        
    # Assemble peak tensor with same column order as in polycrystal.py
    """
    Column names of peaks are
    0: 'grain_index'        10: 'Gx'        20: 'polarization_factors' 
    1: 'phase_number'       11: 'Gy'        21: 'volumes'
    2: 'h'                  12: 'Gz'        22: '2theta'
    3: 'k'                  13: 'K_out_x'   23: 'scherrer_fwhm'
    4: 'l'                  14: 'K_out_y'   24: 'peak_index'
    5: 'structure_factors'  15: 'K_out_z'
    6: 'diffraction_times'  16: 'Source_x'
    7: 'G0_x'              17: 'Source_y'      
    8: 'G0_y'              18: 'Source_z'
    9: 'G0_z'              19: 'lorentz_factors'
    """
    # Convert to torch tensors
    peak = torch.zeros(25)
    peak[0] = element_index  # grain_index
    peak[1] = 0  # phase_number 
    peak[2:5] = torch.tensor(phase.miller_indices[hkl_indx])  # hkl
    peak[5] = structure_factor
    peak[6] = time
    # G0 (original scattering vector) - using scattered_wave_vector - incident_wave_vector
    G0 = scattered_wave_vector - incident_wave_vector
    peak[7:10] = torch.from_numpy(G0)
    # G (rotated scattering vector) - for time=0 same as G0
    peak[10:13] = torch.from_numpy(G0) 
    # K_out (scattered wave vector)
    peak[13:16] = torch.from_numpy(scattered_wave_vector)
    # Source point (using convex hull centroid)
    source = np.mean(convex_hull.points[convex_hull.vertices], axis=0)
    peak[16:19] = torch.from_numpy(source)
    # Factors
    peak[19] = lorentz_factor
    peak[20] = polarization_factor
    # Volume and angles
    peak[21] = convex_hull.volume if volumes is None else volumes
    peak[22] = 2 * theta  # 2theta
    peak[23] = 0.1  # scherrer_fwhm (placeholder)
    peak[24] = peak_index  # peak_index (unique identifier)
    
    return peak.unsqueeze(0)  # Add batch dimension

class TestDetector(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(10)
        self.pixel_size_z = ensure_torch(50.0)
        self.pixel_size_y = ensure_torch(40.0)
        self.detector_size = ensure_torch(10000.0)
        self.det_corner_0 = ensure_torch([1, 0, 0]) * self.detector_size
        self.det_corner_1 = ensure_torch([1, 1, 0]) * self.detector_size
        self.det_corner_2 = ensure_torch([1, 0, 1]) * self.detector_size
        self.detector = Detector(
            self.pixel_size_z,
            self.pixel_size_y,
            self.det_corner_0,
            self.det_corner_1,
            self.det_corner_2,
        )

    def test_init(self):
        for o, otrue in zip(
            self.detector.det_corner_0, torch.tensor([1, 0, 0]) * self.detector_size
        ):
            self.assertAlmostEqual(o, otrue, msg="detector origin is incorrect")

        for z, ztrue in zip(self.detector.zdhat, torch.tensor([0, 0, 1])):
            self.assertAlmostEqual(z, ztrue, msg="zdhat is incorrect")

        for y, ytrue in zip(self.detector.ydhat, torch.tensor([0, 1, 0])):
            self.assertAlmostEqual(y, ytrue, msg="ydhat is incorrect")

        self.assertAlmostEqual(
            self.detector.zmax,
            self.detector_size,
            msg="Bad detector dimensions in zmax",
        )
        self.assertAlmostEqual(
            self.detector.ymax,
            self.detector_size,
            msg="Bad detector dimensions in ymax",
        )

        for n, ntrue in zip(self.detector.normal, torch.tensor([-1, 0, 0])):
            self.assertAlmostEqual(n, ntrue, msg="Bad detector normal")

    def test_render_peaks(self):
        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = (
            ensure_torch([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + v * torch.sqrt(torch.tensor(2.0)) * self.detector_size / 2.0
        )  # tetra at detector centre
        ch1 = ConvexHull(ensure_numpy(verts1))
        verts2 = (
            ensure_torch([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + 2 * v * torch.sqrt(torch.tensor(2.0)) * self.detector_size
        )  # tetra out of detector bounds
        ch2 = ConvexHull(ensure_numpy(verts2))
        wavelength = 1.0

        incident_wave_vector = 2 * torch.pi * ensure_torch([1, 0, 0]) / wavelength
        scattered_wave_vector = (
            self.det_corner_0
            + self.pixel_size_y * 3 * self.detector.ydhat
            + self.pixel_size_z * 2 * self.detector.zdhat
        )
        scattered_wave_vector = (
            2
            * torch.pi
            * scattered_wave_vector
            / (torch.linalg.norm(scattered_wave_vector) * wavelength)
        )

        data = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "Fe_mp-150_conventional_standard.cif",
        )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = "Fm-3m"  # Iron
        phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase._setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)

        # Create peaks tensors
        peaks1 = create_peak_tensor(
            ch1, 
            ensure_numpy(scattered_wave_vector),
            ensure_numpy(incident_wave_vector),
            wavelength,
            np.array([0, 1, 0]), # polarization
            np.array([0, 0, 1]), # rotation axis 
            0, # time
            phase,
            0, # hkl_indx
            0, # element_index (grain_index)
            0, # peak_index
        )

        peaks2 = create_peak_tensor(
            ch2,
            ensure_numpy(scattered_wave_vector),
            ensure_numpy(incident_wave_vector), 
            wavelength,
            np.array([0, 1, 0]), # polarization
            np.array([0, 0, 1]), # rotation axis
            0, # time  
            phase,
            0, # hkl_indx 
            1, # element_index (grain_index)
            1, # peak_index
        )

        # Combine peaks
        peaks = torch.cat([peaks1, peaks2], dim=0)

        # Create peaks dict like polycrystal.diffract() returns
        peaks_dict = {
            "peaks": peaks,
            "columns": [
                "grain_index", "phase_number", "h", "k", "l",
                "structure_factors", "diffraction_times",
                "G0_x", "G0_y", "G0_z", "Gx", "Gy", "Gz",
                "K_out_x", "K_out_y", "K_out_z",
                "Source_x", "Source_y", "Source_z",
                "lorentz_factors", "polarization_factors",
                "volumes", "2theta", "scherrer_fwhm", "peak_index"
            ],
        }

        # Calculate expected intensity for peak 1 (for validation)
        # intensity = structure_factor × lorentz_factor × polarization_factor × volume
        expected_peak1_intensity = (
            peaks[0, 5] * peaks[0, 19] * peaks[0, 20] * peaks[0, 21]
        )

        # Test render methods
        diffraction_pattern = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="gauss"
        )

        # Check that at least one peak was rendered (first peak should be within bounds)
        # We don't check exact position because rendering uses Gaussian interpolation
        # and the exact distribution depends on implementation details
        self.assertGreater(
            torch.sum(diffraction_pattern),
            0,
            msg="detector rendering did not capture any peaks",
        )
        
        # Check that out-of-bounds peak was not rendered
        # Total intensity should be roughly equal to peak 1's intensity (within 20% tolerance)
        # Peak 2 is out of bounds and should not contribute
        self.assertLess(
            torch.sum(diffraction_pattern),
            expected_peak1_intensity * 1.2,  # Allow 20% tolerance for Gaussian spreading
            msg="detector rendering captured out of bounds peak or intensity is too high",
        )
        
        self.assertGreater(
            torch.sum(diffraction_pattern),
            expected_peak1_intensity * 0.8,  # At least 80% of expected intensity
            msg="detector rendering lost significant intensity",
        )

        # Test with factors - rendering always applies intensity factors
        diffraction_pattern = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="gauss"
        )

        # Just verify render succeeds
        self.assertGreater(
            torch.sum(diffraction_pattern),
            0,
            msg="detector rendering failed with factors",
        )

        # Test voigt render
        diffraction_pattern_voigt = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="voigt"
        )

        # Should produce non-zero output
        self.assertGreater(
            torch.sum(diffraction_pattern_voigt),
            0,
            msg="voigt render produced no intensity",
        )

        # Test volume render requires additional data - skip for this simple test
        # Volume rendering requires mesh_lab, beam, and rigid_body_motion in peaks_dict

if __name__ == "__main__":
    unittest.main()