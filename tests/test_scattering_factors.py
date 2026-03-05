"""Tests for the scattering_factors module.

This module tests the Lorentz factor, polarization factor, and Scherrer broadening
calculations used in X-ray diffraction simulations.
"""

import unittest
import torch
import numpy as np
from xrd_simulator.scattering_factors import _lorentz, _polarization, _scherrer

torch.set_default_dtype(torch.float64)


class TestLorentz(unittest.TestCase):
    """Test the Lorentz factor calculation."""

    def test_lorentz_basic(self):
        """Test Lorentz factor for a simple 2theta=90-degree scattering geometry."""
        # Incident beam along x-axis (normalized)
        k_in = torch.tensor([1.0, 0.0, 0.0])
        # Scattered beam at 2theta = 90 degrees (same magnitude as k_in)
        # cos(2theta) = k_in · k_out / |k_in||k_out| = 0 for 2theta=90°
        # So k_out needs to be perpendicular to k_in
        k_out = torch.tensor([[0.0, 1.0, 0.0]])
        # Rotation axis along z - should give eta ≈ 90° (finite Lorentz)
        rot_axis = torch.tensor([0.0, 0.0, 1.0])
        
        lorentz = _lorentz(k_in, k_out, rot_axis)
        
        # For 2theta=90°, theta=45°, sin(2theta)=1
        # eta is the angle between rotation axis and scattering plane normal
        # Lorentz should be finite and positive
        self.assertTrue(torch.isfinite(lorentz).all(), msg="Lorentz factor should be finite")
        self.assertTrue((lorentz > 0).all(), msg="Lorentz factor should be positive")

    def test_lorentz_multiple_reflections(self):
        """Test Lorentz factor for multiple reflections at once."""
        k_in = torch.tensor([1.0, 0.0, 0.0])
        k_out = torch.tensor([
            [0.5, 0.5, 0.0],
            [0.5, 0.3, 0.4],
            [0.7, 0.7, 0.1]
        ])
        rot_axis = torch.tensor([0.0, 0.0, 1.0])
        
        lorentz = _lorentz(k_in, k_out, rot_axis)
        
        self.assertEqual(lorentz.shape, (3,), msg="Should return one Lorentz factor per reflection")
        self.assertTrue(torch.isfinite(lorentz).all(), msg="All Lorentz factors should be finite")

    def test_lorentz_infinity_at_eta_zero(self):
        """Test that Lorentz factor is infinite when eta ≈ 0 (forward scattering)."""
        k_in = torch.tensor([1.0, 0.0, 0.0])
        # Scattered beam nearly parallel to incident beam (eta close to 0)
        k_out = torch.tensor([[1.0, 0.0, 0.1]])  # Small deviation
        # Rotation axis perpendicular to both
        rot_axis = torch.tensor([0.0, 0.0, 1.0])
        
        lorentz = _lorentz(k_in, k_out, rot_axis)
        
        # When eta is close to 0 or theta is close to 0, Lorentz should be inf
        self.assertTrue(torch.isinf(lorentz).any() or lorentz.item() > 100, 
                       msg="Lorentz factor should be very large or infinite for small scattering angles")

    def test_lorentz_single_vector(self):
        """Test Lorentz factor with single k_out vector (not batched)."""
        k_in = torch.tensor([1.0, 0.0, 0.0])
        k_out = torch.tensor([0.5, 0.5, 0.0])  # Single vector, not batched
        rot_axis = torch.tensor([0.0, 0.0, 1.0])
        
        lorentz = _lorentz(k_in, k_out, rot_axis)
        
        # Should return a scalar (0-dim tensor)
        self.assertTrue(lorentz.dim() == 0 or lorentz.numel() == 1, 
                       msg="Single vector input should return scalar")
        self.assertTrue(torch.isfinite(lorentz), msg="Lorentz factor should be finite")


class TestPolarization(unittest.TestCase):
    """Test the polarization factor calculation."""

    def test_polarization_perpendicular(self):
        """Test polarization factor when scattered beam is perpendicular to polarization."""
        # Polarization along z-axis
        pol_vec = torch.tensor([0.0, 0.0, 1.0])
        # Scattered beam in xy-plane (perpendicular to polarization)
        k_out = torch.tensor([[1.0, 1.0, 0.0]])
        
        polarization = _polarization(k_out, pol_vec)
        
        # When scattered beam is perpendicular to polarization, factor should be 1
        self.assertAlmostEqual(polarization.item(), 1.0, places=5,
                              msg="Polarization factor should be 1 for perpendicular scattering")

    def test_polarization_parallel(self):
        """Test polarization factor when scattered beam is parallel to polarization."""
        # Polarization along y-axis
        pol_vec = torch.tensor([0.0, 1.0, 0.0])
        # Scattered beam along y-axis (parallel to polarization)
        k_out = torch.tensor([[0.0, 1.0, 0.0]])
        
        polarization = _polarization(k_out, pol_vec)
        
        # When scattered beam is parallel to polarization, factor should be 0
        self.assertAlmostEqual(polarization.item(), 0.0, places=5,
                              msg="Polarization factor should be 0 for parallel scattering")

    def test_polarization_intermediate(self):
        """Test polarization factor for intermediate angles."""
        pol_vec = torch.tensor([0.0, 1.0, 0.0])
        # Scattered beam at 45 degrees to polarization
        k_out = torch.tensor([[1.0, 1.0, 0.0]])
        
        polarization = _polarization(k_out, pol_vec)
        
        # Factor should be between 0 and 1
        self.assertTrue(0 < polarization.item() < 1,
                       msg="Polarization factor should be between 0 and 1")
        # For 45 degrees: 1 - cos²(45°) = 1 - 0.5 = 0.5
        self.assertAlmostEqual(polarization.item(), 0.5, places=5,
                              msg="Polarization factor should be 0.5 for 45° angle")

    def test_polarization_multiple_reflections(self):
        """Test polarization factor for multiple reflections."""
        pol_vec = torch.tensor([0.0, 0.0, 1.0])
        k_out = torch.tensor([
            [1.0, 0.0, 0.0],  # Perpendicular to polarization
            [0.0, 0.0, 1.0],  # Parallel to polarization
            [1.0, 0.0, 1.0]   # 45 degrees
        ])
        
        polarization = _polarization(k_out, pol_vec)
        
        self.assertEqual(polarization.shape, (3,), msg="Should return one factor per reflection")
        self.assertAlmostEqual(polarization[0].item(), 1.0, places=5)  # Perpendicular
        self.assertAlmostEqual(polarization[1].item(), 0.0, places=5)  # Parallel
        self.assertAlmostEqual(polarization[2].item(), 0.5, places=5)  # 45 degrees

    def test_polarization_single_vector(self):
        """Test polarization factor with single k_out vector."""
        pol_vec = torch.tensor([0.0, 1.0, 0.0])
        k_out = torch.tensor([1.0, 0.0, 0.0])  # Single vector
        
        polarization = _polarization(k_out, pol_vec)
        
        self.assertAlmostEqual(polarization.item(), 1.0, places=5,
                              msg="Should handle single vector input")


class TestScherrer(unittest.TestCase):
    """Test the Scherrer peak broadening calculation."""

    def test_scherrer_basic(self):
        """Test Scherrer FWHM calculation."""
        # Volume of 1 cubic micron
        volumes = torch.tensor([1.0])
        # 2theta = 30 degrees
        two_theta = torch.tensor([np.radians(30)])
        wavelength = 1.54  # Cu K-alpha in Angstroms
        
        fwhm = _scherrer(volumes, two_theta, wavelength)
        
        # FWHM should be positive
        self.assertTrue((fwhm > 0).all(), msg="FWHM should be positive")
        # FWHM should be in radians (small angle for micron-sized crystals)
        self.assertTrue((fwhm < 0.1).all(), msg="FWHM should be small for large crystals")

    def test_scherrer_size_dependence(self):
        """Test that smaller crystals give broader peaks."""
        # Different volumes
        volumes = torch.tensor([1.0, 0.1, 0.01])  # cubic microns
        two_theta = torch.tensor([np.radians(30), np.radians(30), np.radians(30)])
        wavelength = 1.54
        
        fwhm = _scherrer(volumes, two_theta, wavelength)
        
        # Smaller crystals (smaller volumes) should have larger FWHM
        self.assertTrue(fwhm[0] < fwhm[1] < fwhm[2],
                       msg="Smaller crystals should have broader peaks")

    def test_scherrer_angle_dependence(self):
        """Test that higher angles give broader peaks (due to 1/cos(theta))."""
        volumes = torch.tensor([1.0, 1.0, 1.0])
        # Different 2theta angles
        two_theta = torch.tensor([np.radians(20), np.radians(60), np.radians(120)])
        wavelength = 1.54
        
        fwhm = _scherrer(volumes, two_theta, wavelength)
        
        # Higher angles should give larger FWHM (due to 1/cos(theta) factor)
        self.assertTrue(fwhm[0] < fwhm[1] < fwhm[2],
                       msg="Higher scattering angles should give broader peaks")

    def test_scherrer_shape_factor(self):
        """Test that different shape factors give different FWHM."""
        volumes = torch.tensor([1.0])
        two_theta = torch.tensor([np.radians(30)])
        wavelength = 1.54
        
        fwhm_spherical = _scherrer(volumes, two_theta, wavelength, K=0.9)
        fwhm_cubic = _scherrer(volumes, two_theta, wavelength, K=1.0)
        
        # Different K values should give different FWHM
        self.assertNotAlmostEqual(fwhm_spherical.item(), fwhm_cubic.item(),
                                 msg="Different shape factors should give different FWHM")
        # K=1.0 should give larger FWHM than K=0.9
        self.assertTrue(fwhm_cubic > fwhm_spherical,
                       msg="Larger K should give larger FWHM")


if __name__ == "__main__":
    unittest.main()
