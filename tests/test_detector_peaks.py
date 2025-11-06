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
                      element_index, zd, yd, volumes=None):
    """Helper to create a peak tensor with the same structure as polycrystal.diffract()"""
    # Calculate factors that used to be in ScatteringUnit
    k = incident_wave_vector
    kp = scattered_wave_vector
    theta = np.arccos(k.dot(kp) / (np.linalg.norm(k) ** 2)) / 2.0
    korthogonal = kp - (k * kp.dot(k) / (np.linalg.norm(k) ** 2))
    eta = np.arccos(rotation_axis.dot(korthogonal) / np.linalg.norm(korthogonal))

    # Calculate polarization
    khatp = kp / np.linalg.norm(kp)
    polarization_factor = 1 - np.dot(incident_polarization_vector, khatp) ** 2

    # Calculate lorentz
    tol = 0.5
    if (abs(np.degrees(eta)) < tol or abs(np.degrees(eta)) > 180 - tol 
            or np.degrees(theta) < tol):
        lorentz_factor = float('inf')
    else:
        lorentz_factor = 1.0 / (np.sin(2 * theta) * abs(np.sin(eta)))

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
    4: 'l'                  14: 'K_out_y'   
    5: 'structure_factors'  15: 'K_out_z'
    6: 'diffraction_times'  16: 'Source_x'
    7: 'G0_x'              17: 'Source_y'      
    8: 'G0_y'              18: 'Source_z'
    9: 'G0_z'              19: 'lorentz_factors'
    """
    # Convert to torch tensors
    peak = torch.zeros(24)
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
        phase.setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)

        # Create peaks tensors
        zd1, yd1, _ = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0, keepdim=True)
            )[0]
        )
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
            0, # element_index
            ensure_numpy(zd1),
            ensure_numpy(yd1)
        )

        zd2, yd2, _ = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts2.mean(dim=0)[None, :]
            )[0]
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
            1, # element_index
            ensure_numpy(zd2),
            ensure_numpy(yd2)
        )

        # Combine peaks
        peaks = torch.cat([peaks1, peaks2], dim=0)
        
        # Create convex hulls list
        convex_hulls = [ch1, ch2]

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
                "volumes", "2theta", "scherrer_fwhm"
            ],
            "convex_hulls": convex_hulls,
            "scattered_vectors": peaks[:, 13:16],
            "incident_vector": incident_wave_vector,
            "wavelength": wavelength,
            "rotation_axis": ensure_torch([0, 0, 1])
        }

        # Test render methods
        diffraction_pattern = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="gauss"
        )

        # Test expected peak position and intensity
        expected_z_pixel = int(self.detector_size / (2 * self.pixel_size_z)) + 2
        expected_y_pixel = int(self.detector_size / (2 * self.pixel_size_y)) + 3

        dy = self.detector._point_spread_kernel_shape[0]
        dz = self.detector._point_spread_kernel_shape[1]
        active_det_part = diffraction_pattern[0,
            expected_z_pixel - dy + 1 : expected_z_pixel + dy,
            expected_y_pixel - dz + 1 : expected_y_pixel + dz,
        ]

        self.assertAlmostEqual(
            torch.sum(active_det_part),
            ch1.volume,
            msg="detector rendering did not capture peak volume",
        )
        self.assertAlmostEqual(
            torch.sum(diffraction_pattern),
            ch1.volume,
            msg="detector rendering captured out of bounds peak",
        )

        # Test with factors
        diffraction_pattern = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="gauss"
        )

        self.assertTrue(
            diffraction_pattern[0, expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not apply intensity factors",
        )

        # Test voigt render
        diffraction_pattern = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="voigt"
        )

        # Should still sum to volume
        self.assertAlmostEqual(
            torch.sum(diffraction_pattern),
            ch1.volume,
            msg="voigt render changed total intensity",
        )

        # Test volume render
        diffraction_pattern = self.detector.render(
            peaks_dict,
            frames_to_render=1,
            method="volumes"
        )

        # Should still sum to volume
        self.assertAlmostEqual(
            torch.sum(diffraction_pattern),
            ch1.volume,
            msg="volume render changed total intensity",
        )

if __name__ == "__main__":
    unittest.main()