import unittest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import os
import torch
from xrd_simulator.detector import Detector
from xrd_simulator.phase import Phase
from xrd_simulator.beam import Beam
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.motion import RigidBodyMotion
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
            det_corner_0=self.det_corner_0,
            det_corner_1=self.det_corner_1,
            det_corner_2=self.det_corner_2,
            pixel_size=(self.pixel_size_z, self.pixel_size_y),
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

    def test_save_and_load(self):
        """Test detector serialization and deserialization."""
        path = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "my_detector_test"
        )
        
        # Save detector
        self.detector.save(path)
        
        # Load detector
        loaded_detector = Detector.load(path + ".det")
        
        # Verify loaded detector matches original
        self.assertAlmostEqual(
            loaded_detector.pixel_size_z.item(), 
            self.detector.pixel_size_z.item(),
            msg="Pixel size z corrupted on save/load"
        )
        self.assertAlmostEqual(
            loaded_detector.pixel_size_y.item(), 
            self.detector.pixel_size_y.item(),
            msg="Pixel size y corrupted on save/load"
        )
        self.assertTrue(
            torch.allclose(loaded_detector.det_corner_0, self.detector.det_corner_0),
            msg="det_corner_0 corrupted on save/load"
        )
        self.assertTrue(
            torch.allclose(loaded_detector.normal, self.detector.normal),
            msg="normal corrupted on save/load"
        )
        
        # Clean up
        os.remove(path + ".det")

    def test_render(self):
        """Test all rendering methods: micro, nano, macro, and auto.
        
        Tests:
        - micro: Gaussian profiles for medium grains
        - nano: Airy disk patterns for small crystallites  
        - macro: Volume projection for large grains
        - auto: Automatic method selection based on grain size
        - Infinite Lorentz factor handling (should warn and skip)
        """
        import warnings
        
        # --- Setup common test data ---
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
            2 * torch.pi * scattered_wave_vector
            / (torch.linalg.norm(scattered_wave_vector) * wavelength)
        )

        zd1,yd1 = tuple(self.detector.get_intersection(scattered_wave_vector,verts1.mean(axis=0)[np.newaxis,:])[0])
        zd2,yd2 = tuple(self.detector.get_intersection(scattered_wave_vector,verts2.mean(axis=0)[np.newaxis,:])[0])
        
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
            np.array([0, 1, 0]),  # polarization
            np.array([0, 0, 1]),  # rotation axis 
            0,  # time
            phase,
            0,  # hkl_indx
            0,  # element_index
            0,  # peak_index
        )

        peaks2 = create_peak_tensor(
            ch2,
            ensure_numpy(scattered_wave_vector),
            ensure_numpy(incident_wave_vector), 
            wavelength,
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            0,
            phase,
            0,
            1,
            1,
        )

        peaks = torch.cat([peaks1, peaks2], dim=0)
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

        expected_peak1_intensity = (
            peaks[0, 5] * peaks[0, 19] * peaks[0, 20] * peaks[0, 21]
        )

        # --- Test micro rendering ---
        diffraction_pattern, _ = self.detector.render(
            peaks_dict, frames_to_render=1, method="micro"
        )
        self.assertGreater(
            torch.sum(diffraction_pattern), 0,
            msg="micro render did not capture any peaks",
        )
        # Out-of-bounds peak should not be rendered
        self.assertLess(
            torch.sum(diffraction_pattern),
            expected_peak1_intensity * 1.2,
            msg="micro render captured out of bounds peak",
        )
        self.assertGreater(
            torch.sum(diffraction_pattern),
            expected_peak1_intensity * 0.8,
            msg="micro render lost significant intensity",
        )

        # --- Test nano rendering ---
        diffraction_pattern_nano, _ = self.detector.render(
            peaks_dict, frames_to_render=1, method="nano"
        )
        self.assertGreater(
            torch.sum(diffraction_pattern_nano), 0,
            msg="nano render produced no intensity",
        )

        # --- Test macro rendering ---
        # Macro requires mesh_lab, beam, and rigid_body_motion
        mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.linalg.norm(x) - 500.0,
            bounding_radius=600.0,
            max_cell_circumradius=200.0,
        )
        beam_vertices = np.array([
            [-20000, -10000, -10000], [-20000, 10000, -10000],
            [-20000, 10000, 10000], [-20000, -10000, 10000],
            [20000, -10000, -10000], [20000, 10000, -10000],
            [20000, 10000, 10000], [20000, -10000, 10000],
        ], dtype=float)
        beam = Beam(beam_vertices, np.array([1, 0, 0]), wavelength, np.array([0, 1, 0]))
        motion = RigidBodyMotion(np.array([0, 0, 1]), np.radians(10), np.array([0, 0, 0]))

        peaks_dict_macro = {
            "peaks": peaks1.clone(),
            "columns": peaks_dict["columns"],
            "beam": beam,
            "mesh_lab": mesh,
            "rigid_body_motion": motion,
        }
        diffraction_pattern_macro, _ = self.detector.render(
            peaks_dict_macro, frames_to_render=1, method="macro"
        )
        self.assertEqual(
            diffraction_pattern_macro.dim(), 3,
            msg="macro render should return 3D tensor"
        )

        # --- Test auto rendering with grains from all size regimes ---
        # Thresholds (matching detector.render defaults):
        #   nano:  volume < 0.001 µm³ (< ~100 nm grain diameter)
        #   micro: 0.001 µm³ <= volume < pixel_size³ (~100 nm to ~40 µm)
        #   macro: volume >= pixel_size³ (> ~40 µm, uses volume projection)
        pixel_size = min(self.pixel_size_y.item(), self.pixel_size_z.item())
        macro_limit = pixel_size**3  # 64000 µm³ for 40 µm pixels
        
        # Create peaks with volumes spanning all three physical size scales:
        # - Nanometer-scale crystallites (< 100 nm)
        # - Micrometer-scale grains (~1-40 µm)
        # - Millimeter-scale grains (~1 mm)
        peaks_auto = peaks1.clone().repeat(6, 1)
        peaks_auto[0, 21] = 0.0001      # nano: ~(46 nm)³
        peaks_auto[1, 21] = 0.0005      # nano: ~(79 nm)³
        peaks_auto[2, 21] = 100.0       # micro: ~(4.6 µm)³
        peaks_auto[3, 21] = 10000.0     # micro: ~(21.5 µm)³
        peaks_auto[4, 21] = macro_limit * 2    # macro: ~(50 µm)³
        peaks_auto[5, 21] = 1e9         # macro: (1 mm)³ - true millimeter grain
        for i in range(6):
            peaks_auto[i, 24] = i  # unique peak_index
        
        peaks_dict_auto = {
            "peaks": peaks_auto,
            "columns": peaks_dict["columns"],
            "beam": beam,
            "mesh_lab": mesh,
            "rigid_body_motion": motion,
        }
        diffraction_pattern_auto, _ = self.detector.render(
            peaks_dict_auto, frames_to_render=1, method="auto", verbose=False
        )
        self.assertEqual(diffraction_pattern_auto.dim(), 3)
        self.assertGreater(torch.sum(diffraction_pattern_auto), 0,
            msg="auto render produced no intensity")

        # --- Test infinite Lorentz factor handling ---
        peak_inf = peaks1.clone()
        peak_inf[0, 19] = float('inf')  # Set infinite Lorentz factor
        peaks_dict_inf = {
            "peaks": peak_inf,
            "columns": peaks_dict["columns"],
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diffraction_pattern_inf, _ = self.detector.render(
                peaks_dict_inf, frames_to_render=1, method="micro"
            )
            self.assertEqual(len(w), 1, msg="Expected warning for infinite Lorentz")
            self.assertIn("infinite Lorentz", str(w[0].message))
        
        self.assertEqual(
            torch.sum(diffraction_pattern_inf), 0,
            msg="Pattern should be empty when all peaks have infinite Lorentz"
        )

    def test_contains(self):
        """Test that detector.contains correctly identifies in-bounds and out-of-bounds coordinates."""
        # Test point inside detector bounds
        zd_inside = self.detector_size / 2.0
        yd_inside = self.detector_size / 2.0
        self.assertTrue(
            self.detector.contains(zd_inside, yd_inside),
            msg="Detector should contain point at center"
        )

        # Test point at origin (edge case)
        self.assertTrue(
            self.detector.contains(torch.tensor(0.0), torch.tensor(0.0)),
            msg="Detector should contain point at origin"
        )

        # Test point outside detector (negative coordinates)
        self.assertFalse(
            self.detector.contains(torch.tensor(-100.0), self.detector_size / 2.0),
            msg="Detector should not contain negative z coordinate"
        )
        self.assertFalse(
            self.detector.contains(self.detector_size / 2.0, torch.tensor(-100.0)),
            msg="Detector should not contain negative y coordinate"
        )

        # Test point outside detector (beyond max)
        self.assertFalse(
            self.detector.contains(self.detector_size * 2, self.detector_size / 2.0),
            msg="Detector should not contain z coordinate beyond zmax"
        )
        self.assertFalse(
            self.detector.contains(self.detector_size / 2.0, self.detector_size * 2),
            msg="Detector should not contain y coordinate beyond ymax"
        )

    def test_get_intersection(self):
        """Test ray-detector intersection calculation."""
        # Central normal-aligned ray from origin
        ray_direction = torch.tensor([2.23, 0.0, 0.0])
        source_point = torch.tensor([0.0, 0.0, 0.0])

        result = self.detector._get_intersection(ray_direction, source_point)
        zd, yd = result[0, 0], result[0, 1]
        
        self.assertAlmostEqual(
            zd.item(), 0, places=5,
            msg="Central detector-normal aligned ray should intersect at z=0"
        )
        self.assertAlmostEqual(
            yd.item(), 0, places=5,
            msg="Central detector-normal aligned ray should intersect at y=0"
        )

        # Translate the source point
        source_point = (
            self.detector.ydhat * self.pixel_size_y 
            - self.detector.zdhat * 2 * self.pixel_size_z
        )
        result = self.detector._get_intersection(ray_direction, source_point)
        zd, yd = result[0, 0], result[0, 1]
        
        self.assertAlmostEqual(
            zd.item(), -2 * self.pixel_size_z.item(), places=3,
            msg="Translated ray should intersect at correct z"
        )
        self.assertAlmostEqual(
            yd.item(), self.pixel_size_y.item(), places=3,
            msg="Translated ray should intersect at correct y"
        )

        # Test multiple rays at once (batch)
        ray_directions = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.1]
        ])
        source_points = torch.zeros(3, 3)
        
        results = self.detector._get_intersection(ray_directions, source_points)
        self.assertEqual(results.shape, (3, 3), msg="Batch intersection should return (N, 3)")

    def test_get_wrapping_cone(self):
        """Test computation of cone wrapping the detector."""
        wavelength = 1.0
        k = 2 * torch.pi * torch.tensor([1.0, 0.0, 0.0]) / wavelength
        source_point = (
            self.detector.zdhat + self.detector.ydhat
        ) * self.detector_size / 2.0
        
        opening_angle = self.detector._get_wrapping_cone(k, source_point)

        # Calculate expected angle
        normed_det_center = (source_point + self.det_corner_0) / torch.linalg.norm(
            source_point + self.det_corner_0
        )
        normed_det_origin = self.det_corner_0 / torch.linalg.norm(self.det_corner_0)
        expected_angle = torch.arccos(torch.dot(normed_det_center, normed_det_origin)) / 2.0

        self.assertAlmostEqual(
            opening_angle.item(), expected_angle.item(), places=4,
            msg="Detector-centered wrapping cone has incorrect opening angle"
        )

        # Test off-center source point gives larger cone
        source_point_offset = source_point.clone()
        source_point_offset -= self.detector.zdhat * 10 * self.pixel_size_z
        source_point_offset -= self.detector.ydhat * 10 * self.pixel_size_y
        opening_angle_offset = self.detector._get_wrapping_cone(k, source_point_offset)
        
        self.assertGreaterEqual(
            opening_angle_offset.item(), expected_angle.item(),
            msg="Off-centered source should give larger or equal cone angle"
        )

    def test_lorentz_infinity_handling_nano(self):
        """Test that rendering properly handles infinite Lorentz factors.
        
        When eta ≈ 0 (grazing geometry), the Lorentz factor becomes infinite.
        The renderer should skip peaks with infinite Lorentz factors and emit a warning,
        as these represent geometrically impossible reflections.
        """
        import warnings
        
        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        
        # Create tiny scattering unit at detector center
        verts = (
            ensure_torch([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]) * 0.0000001
            + v * torch.sqrt(torch.tensor(2.0)) * self.detector_size / 2.0
            + 0.001 * self.detector.ydhat
            + 0.001 * self.detector.zdhat
        )
        ch = ConvexHull(ensure_numpy(verts))

        wavelength = 1.0
        incident_wave_vector = 2 * torch.pi * ensure_torch([1, 0, 0]) / wavelength
        # Scattered wave nearly parallel to incident (small scattering angle -> infinite Lorentz)
        scattered_wave_vector = 2 * torch.pi * ensure_torch([1, 0, 0.1]) / (
            torch.sqrt(torch.tensor(0.1 * 0.1 + 1.0)) * wavelength
        )

        data = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "Fe_mp-150_conventional_standard.cif"
        )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = "Fm-3m"
        phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase._setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)

        # Create peak with infinite Lorentz factor
        peak = create_peak_tensor(
            ch,
            ensure_numpy(scattered_wave_vector),
            ensure_numpy(incident_wave_vector),
            wavelength,
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            0,
            phase,
            0,
            0,
            0,
        )
        # Manually set Lorentz factor to infinity to test handling
        peak[0, 19] = float('inf')

        peaks_dict = {
            "peaks": peak,
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

        # Rendering should emit a warning about skipped peaks
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            diffraction_pattern_micro, _ = self.detector.render(
                peaks_dict,
                frames_to_render=1,
                method="micro"
            )
            
            # Check that a warning was issued
            self.assertEqual(len(w), 1, msg="Expected exactly one warning for infinite Lorentz")
            self.assertIn("infinite Lorentz", str(w[0].message))
            self.assertIn("Skipping", str(w[0].message))
        
        # The pattern should be empty since the only peak was filtered out
        self.assertEqual(
            diffraction_pattern_micro.dim(), 3,
            msg="Micro render should return 3D tensor"
        )
        self.assertEqual(
            torch.sum(diffraction_pattern_micro), 0,
            msg="Pattern should be empty when all peaks have infinite Lorentz"
        )
        
        # Test nano method as well - should also warn and produce empty output
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            diffraction_pattern_nano, _ = self.detector.render(
                peaks_dict,
                frames_to_render=1,
                method="nano"
            )
            
            self.assertEqual(len(w), 1, msg="Expected warning for nano render too")
        
        self.assertEqual(
            diffraction_pattern_nano.dim(), 3,
            msg="Nano render should return 3D tensor"
        )
        self.assertEqual(
            torch.sum(diffraction_pattern_nano), 0,
            msg="Pattern should be empty when all peaks have infinite Lorentz"
        )

if __name__ == "__main__":
    unittest.main()