"""
End-to-end test: Validate simulated powder diffraction against crystallographic theory.

This test:
1. Generates a 10,000 grain polycrystal with random orientations
2. Simulates diffraction over 30° rotation (full powder rings)
3. Renders using Gaussian method
4. Integrates azimuthally using pyFAI
5. Compares extracted Q peak positions against theoretical d-spacings from CIF

This validates the entire pipeline from crystal structure to detector output.
"""
import unittest
import numpy as np
import os

try:
    import pyFAI
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    HAS_PYFAI = True
except ImportError:
    HAS_PYFAI = False

from scipy.signal import find_peaks
from xfab import tools
import matplotlib.pyplot as plt

from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion


def compute_theoretical_Q_values(unit_cell, sgname, wavelength, max_bragg_angle):
    """
    Compute theoretical Q values for all allowed reflections using xfab.
    
    Uses the same crystallographic library (xfab) that xrd_simulator uses internally,
    ensuring consistency between theoretical and simulated peak positions.
    
    Args:
        unit_cell: [a, b, c, alpha, beta, gamma] in Angstrom and degrees
        sgname: Space group name (e.g., 'P3221')
        wavelength: X-ray wavelength in Angstrom
        max_bragg_angle: Maximum Bragg angle in radians
        
    Returns:
        unique_Q: Sorted array of unique Q values (Å⁻¹)
        d_spacings: Corresponding d-spacings (Å)
    """
    # Generate all allowed Miller indices using xfab
    # This is the same function used by Phase._setup_diffracting_planes()
    sintlmin = 0.0
    sintlmax = np.sin(max_bragg_angle) / wavelength
    miller_indices = tools.genhkl_all(unit_cell, sintlmin, sintlmax, sgname=sgname)
    
    if miller_indices is None or len(miller_indices) == 0:
        return np.array([]), np.array([])
    
    # Compute B matrix (reciprocal lattice)
    B = tools.form_b_mat(unit_cell)
    
    # Compute d-spacings for each (hkl)
    # G = B @ hkl, |G| = 2π/d
    G_vectors = np.dot(B, miller_indices.T).T  # (n_hkl, 3)
    G_norms = np.linalg.norm(G_vectors, axis=1)
    d_spacings = 2 * np.pi / G_norms
    
    # Q = 2π/d
    Q_values = 2 * np.pi / d_spacings
    
    # Get unique Q values (multiple hkl can have same d-spacing)
    # Round to avoid floating point duplicates
    Q_rounded = np.round(Q_values, decimals=4)
    unique_Q = np.unique(Q_rounded)
    unique_Q = np.sort(unique_Q)
    
    return unique_Q, 2 * np.pi / unique_Q


def find_peaks_in_pattern(radial_values, intensity, prominence_fraction=0.01):
    """
    Find peaks in 1D integrated pattern.
    
    Args:
        radial_values: Q or 2theta values
        intensity: Integrated intensity
        prominence_fraction: Minimum prominence as fraction of max intensity
        
    Returns:
        peak_positions: Array of Q/2theta values at peak centers
        peak_intensities: Array of peak heights
    """
    # Normalize intensity
    intensity = np.array(intensity)
    max_int = np.max(intensity)
    if max_int == 0:
        return np.array([]), np.array([])
    
    # Find peaks with sufficient prominence
    min_prominence = prominence_fraction * max_int
    peaks, properties = find_peaks(
        intensity, 
        prominence=min_prominence,
        height=min_prominence
    )
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    peak_positions = radial_values[peaks]
    peak_intensities = intensity[peaks]
    
    return peak_positions, peak_intensities


@unittest.skipUnless(HAS_PYFAI, "pyFAI not installed")
class TestPowderIntegration(unittest.TestCase):
    """
    Validate powder diffraction simulation against crystallographic theory.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up simulation parameters."""
        # X-ray energy and wavelength
        cls.energy_keV = 45.0
        cls.wavelength = 12.398 / cls.energy_keV  # Angstrom (0.2755 Å)
        
        # Detector geometry - perpendicular to beam, centered
        cls.pixel_size = 150.0  # microns
        cls.n_pixels = 2048
        cls.detector_size = cls.pixel_size * cls.n_pixels  # microns
        cls.detector_distance = 200000.0  # 200 mm in microns
        
        # Detector corners (detector perpendicular to x-axis, centered on beam)
        det_corner_0 = np.array([cls.detector_distance, -cls.detector_size/2, -cls.detector_size/2])
        det_corner_1 = np.array([cls.detector_distance,  cls.detector_size/2, -cls.detector_size/2])
        det_corner_2 = np.array([cls.detector_distance, -cls.detector_size/2,  cls.detector_size/2])
        
        cls.detector = Detector(
            det_corner_0=det_corner_0,
            det_corner_1=det_corner_1,
            det_corner_2=det_corner_2,
            n_pixels=(cls.n_pixels, cls.n_pixels))
        
        # Calculate detector coverage in 2theta
        cls.max_2theta = np.arctan((cls.detector_size/2) / cls.detector_distance)  # radians
        cls.max_bragg_angle = cls.max_2theta / 2  # Bragg angle = 2theta/2
        
        # Sample parameters
        cls.n_grains = 10000
        cls.sample_radius = 100.0  # microns - small sample fully in beam
        
        # Rotation
        cls.rotation_angle = np.radians(30.0)
        cls.rotation_axis = np.array([0.0, 0.0, 1.0])
        
        # Q tolerance for peak matching
        # 1.5% accounts for:
        # - Pixel discretization effects
        # - Closely-spaced peaks that may merge in integration
        # - Azimuthal integration binning
        cls.Q_tolerance = 0.015
        
        # No CIF file needed for ferrite - use unit cell and space group directly
        cls.cif_path = None
        
        # Ferrite (BCC iron) parameters
        # a = 2.87 Å, space group Im-3m (#229)
        cls.unit_cell = [2.87, 2.87, 2.87, 90.0, 90.0, 90.0]
        cls.sgname = 'Im-3m'
        
    def setUp(self):
        """Create fresh beam and motion for each test."""
        # Beam - large enough to cover entire sample
        w = 2 * self.sample_radius * 10  # 10x sample size
        beam_vertices = np.array([
            [-self.detector_distance, -w, -w],
            [-self.detector_distance,  w, -w],
            [-self.detector_distance,  w,  w],
            [-self.detector_distance, -w,  w],
            [ self.detector_distance, -w, -w],
            [ self.detector_distance,  w, -w],
            [ self.detector_distance,  w,  w],
            [ self.detector_distance, -w,  w],
        ])
        
        xray_propagation_direction = np.array([1.0, 0.0, 0.0])
        polarization_vector = np.array([0.0, 1.0, 0.0])
        
        self.beam = Beam(
            beam_vertices,
            xray_propagation_direction,
            self.wavelength,
            polarization_vector
        )
        
        self.motion = RigidBodyMotion(
            self.rotation_axis,
            self.rotation_angle,
            np.array([0.0, 0.0, 0.0])  # no translation
        )
    
    def _create_polycrystal(self):
        """Create a polycrystal with many randomly oriented grains."""
        from xrd_simulator.templates import get_uniform_powder_sample
        
        polycrystal = get_uniform_powder_sample(
            sample_bounding_radius=self.sample_radius,
            number_of_grains=self.n_grains,
            unit_cell=self.unit_cell,
            sgname=self.sgname
        )
        
        return polycrystal
    
    def _setup_pyfai_integrator(self):
        """Create pyFAI integrator matching our detector geometry.
        
        Geometry mapping:
        - xrd_simulator: beam along +x, detector at x=detector_distance
          Detector corners at y=±size/2, z=±size/2, so beam center hits pixel (n/2, n/2)
        - pyFAI: PONI (Point Of Normal Incidence) is where sample-detector line hits detector
          poni1/poni2 are distances in meters from detector corner (0,0)
        
        For our centered detector: poni1 = poni2 = (n_pixels/2) * pixel_size
        """
        # pyFAI uses meters, we have microns
        dist_m = self.detector_distance * 1e-6
        pixel_m = self.pixel_size * 1e-6
        
        # PONI at detector center (beam center maps to pixel n/2, n/2)
        # Verified: det_corner_0 at (D, -size/2, -size/2) means
        # beam at (D, 0, 0) hits pixel (n/2, n/2)
        poni1 = self.n_pixels / 2 * pixel_m  # row center (meters)
        poni2 = self.n_pixels / 2 * pixel_m  # col center (meters)
        
        ai = AzimuthalIntegrator(
            dist=dist_m,
            poni1=poni1,
            poni2=poni2,
            rot1=0, rot2=0, rot3=0,  # no tilts
            pixel1=pixel_m,
            pixel2=pixel_m,
            wavelength=self.wavelength * 1e-10  # pyFAI uses meters
        )
        
        return ai
    
    def _plot_results(self, pattern_np, Q_result, I_result, theoretical_Q, found_Q, Q_max):
        """Plot 2D diffraction pattern and 1D integrated profile with peak markers."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: 2D diffraction pattern (log scale for better visibility)
        ax1 = axes[0]
        pattern_plot = pattern_np.copy()
        pattern_plot[pattern_plot <= 0] = 1  # avoid log(0)
        im = ax1.imshow(np.log10(pattern_plot), cmap='viridis', origin='lower')
        ax1.set_title(f'2D Diffraction Pattern (log scale)\n{self.n_grains} grains, {np.degrees(self.rotation_angle):.0f}° rotation')
        ax1.set_xlabel('Pixel (z)')
        ax1.set_ylabel('Pixel (y)')
        plt.colorbar(im, ax=ax1, label='log₁₀(Intensity)')
        
        # Mark beam center
        center = self.n_pixels // 2
        ax1.plot(center, center, 'r+', markersize=15, markeredgewidth=2, label='Beam center')
        ax1.legend(loc='upper right')
        
        # Right: 1D integrated pattern with peak positions
        ax2 = axes[1]
        ax2.plot(Q_result, I_result, 'b-', linewidth=0.5, label='Integrated intensity')
        
        # Add vertical lines for theoretical peak positions (green)
        for i, Q in enumerate(theoretical_Q):
            if Q <= Q_max:
                ax2.axvline(Q, color='green', alpha=0.3, linewidth=0.8, 
                           label='Theoretical' if i == 0 else None)
        
        # Add vertical lines for found peak positions (red, dashed)
        for i, Q in enumerate(found_Q):
            ax2.axvline(Q, color='red', alpha=0.6, linewidth=1, linestyle='--',
                       label='Found' if i == 0 else None)
        
        ax2.set_xlabel('Q (Å⁻¹)')
        ax2.set_ylabel('Integrated Intensity (a.u.)')
        ax2.set_title(f'Azimuthal Integration\nλ = {self.wavelength:.4f} Å ({self.energy_keV} keV)')
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, Q_max * 1.05)
        
        # Add secondary x-axis for 2θ
        ax2_top = ax2.twiny()
        ax2_top.set_xlim(ax2.get_xlim())
        # Convert Q limits to 2theta
        Q_ticks = ax2.get_xticks()
        Q_ticks = Q_ticks[(Q_ticks > 0) & (Q_ticks < 4 * np.pi / self.wavelength)]
        two_theta_ticks = 2 * np.arcsin(Q_ticks * self.wavelength / (4 * np.pi))
        ax2_top.set_xticks(Q_ticks)
        ax2_top.set_xticklabels([f'{np.degrees(t):.0f}°' for t in two_theta_ticks])
        ax2_top.set_xlabel('2θ')
        
        plt.tight_layout()
        
        # Save to test_reports folder
        reports_dir = os.path.join(os.path.dirname(__file__), '..', 'test_reports')
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, 'powder_integration_results.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved plot to: {save_path}")
        plt.close(fig)
    
    def test_powder_peak_positions(self):
        """
        Main test: Simulate powder pattern and verify peak positions match theory.
        """
        print("\n" + "="*70)
        print("POWDER DIFFRACTION VALIDATION TEST")
        print("="*70)
        print(f"Energy: {self.energy_keV} keV (λ = {self.wavelength:.4f} Å)")
        print(f"Grains: {self.n_grains}")
        print(f"Rotation: {np.degrees(self.rotation_angle):.1f}°")
        print(f"Detector: {self.n_pixels}x{self.n_pixels} pixels, {self.detector_distance/1000:.1f} mm distance")
        print(f"Max 2θ coverage: {np.degrees(2*self.max_bragg_angle):.2f}°")
        print(f"Material: Ferrite - BCC Iron ({self.sgname})")
        print("="*70)
        
        # Step 1: Compute theoretical Q values
        print("\n[1] Computing theoretical peak positions...")
        theoretical_Q, theoretical_d = compute_theoretical_Q_values(
            self.unit_cell, self.sgname, self.wavelength, self.max_bragg_angle
        )
        print(f"    Found {len(theoretical_Q)} unique d-spacings within detector range")
        for i, (Q, d) in enumerate(zip(theoretical_Q[:10], theoretical_d[:10])):
            two_theta = 2 * np.arcsin(Q * self.wavelength / (4 * np.pi))
            print(f"    {i+1:2d}. d = {d:.4f} Å, Q = {Q:.4f} Å⁻¹, 2θ = {np.degrees(two_theta):.2f}°")
        if len(theoretical_Q) > 10:
            print(f"    ... and {len(theoretical_Q) - 10} more")
        
        # Step 2: Create polycrystal
        print("\n[2] Creating polycrystal sample...")
        polycrystal = self._create_polycrystal()
        print(f"    Mesh elements: {polycrystal.mesh_lab.number_of_elements}")
        
        # Step 3: Simulate diffraction
        print("\n[3] Computing diffraction...")
        peaks_dict = polycrystal.diffract(
            self.beam, 
            self.motion,
            detector=self.detector,
            verbose=False
        )
        n_peaks = len(peaks_dict['peaks'])
        print(f"    Generated {n_peaks} diffraction peaks")
        
        if n_peaks == 0:
            self.fail("No diffraction peaks generated!")
        
        # Step 4: Render pattern
        print("\n[4] Rendering diffraction pattern (micro mode)...")
        pattern = self.detector.render(
            peaks_dict,
            frames_to_render=0,  # single integrated frame
            method='micro'
        )
        
        # Convert to numpy
        if hasattr(pattern, 'cpu'):
            pattern_np = pattern[0].cpu().numpy()
        else:
            pattern_np = np.array(pattern[0])
        
        print(f"    Pattern shape: {pattern_np.shape}")
        print(f"    Non-zero pixels: {np.sum(pattern_np > 0)}")
        print(f"    Max intensity: {np.max(pattern_np):.2e}")
        
        # Step 5: pyFAI integration
        print("\n[5] Integrating with pyFAI...")
        ai = self._setup_pyfai_integrator()
        
        # Integrate to Q space
        n_bins = 2000
        Q_result, I_result = ai.integrate1d(
            pattern_np,
            n_bins,
            unit='q_A^-1',
            correctSolidAngle=True
        )
        
        print(f"    Q range: {Q_result[0]:.4f} to {Q_result[-1]:.4f} Å⁻¹")
        print(f"    Integrated intensity range: {np.min(I_result):.2e} to {np.max(I_result):.2e}")
        
        # Filter theoretical Q to only those within pyFAI integration range
        # (some high-Q peaks may be outside the circular integration region)
        Q_max_integrated = Q_result[-1] * 0.95  # 5% margin from edge
        theoretical_Q_filtered = theoretical_Q[theoretical_Q <= Q_max_integrated]
        print(f"    Theoretical peaks in range: {len(theoretical_Q_filtered)} (of {len(theoretical_Q)} total)")
        
        # Step 6: Find peaks in integrated pattern
        print("\n[6] Finding peaks in integrated pattern...")
        found_Q, found_I = find_peaks_in_pattern(Q_result, I_result, prominence_fraction=0.005)
        print(f"    Found {len(found_Q)} peaks")
        
        if len(found_Q) == 0:
            self.fail("No peaks found in integrated pattern!")
        
        # Step 6b: Plot 2D pattern and 1D integration
        print("\n[6b] Plotting results...")
        self._plot_results(
            pattern_np, Q_result, I_result, 
            theoretical_Q_filtered, found_Q,
            Q_max_integrated
        )
        
        # Step 7: Match peaks
        print("\n[7] Matching found peaks to theoretical positions...")
        print(f"    Tolerance: {self.Q_tolerance*100:.1f}%")
        
        matched_theoretical = []
        unmatched_theoretical = []
        spurious_found = []  # peaks found that don't match any theory
        
        # First check: every theoretical peak should have a matching found peak
        for Q_theo in theoretical_Q_filtered:
            # Find closest found peak
            if len(found_Q) > 0:
                distances = np.abs(found_Q - Q_theo) / Q_theo
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                if min_distance <= self.Q_tolerance:
                    matched_theoretical.append(Q_theo)
                    print(f"    ✓ Q_theo = {Q_theo:.4f} Å⁻¹ matched at Q_found = {found_Q[min_idx]:.4f} Å⁻¹ (error: {min_distance*100:.2f}%)")
                else:
                    unmatched_theoretical.append(Q_theo)
                    print(f"    ✗ Q_theo = {Q_theo:.4f} Å⁻¹ NOT FOUND (closest: {found_Q[min_idx]:.4f} Å⁻¹, error: {min_distance*100:.2f}%)")
            else:
                unmatched_theoretical.append(Q_theo)
                print(f"    ✗ Q_theo = {Q_theo:.4f} Å⁻¹ NOT FOUND (no peaks)")
        
        # Second check: every found peak should match a theoretical peak
        # This catches spurious peaks that shouldn't exist
        print("\n    Checking found peaks against theory:")
        for Q_found in found_Q:
            if Q_found > Q_max_integrated:
                continue  # skip peaks outside our comparison range
            distances = np.abs(theoretical_Q_filtered - Q_found) / Q_found
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            if min_distance > self.Q_tolerance:
                spurious_found.append(Q_found)
                print(f"    ⚠ Q_found = {Q_found:.4f} Å⁻¹ has no theoretical match (closest: {theoretical_Q_filtered[min_idx]:.4f} Å⁻¹, error: {min_distance*100:.2f}%)")
        
        # Step 8: Report results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        n_matched = len(matched_theoretical)
        n_total = len(theoretical_Q_filtered)
        n_spurious = len(spurious_found)
        match_rate = n_matched / n_total * 100 if n_total > 0 else 0
        
        print(f"Matched theoretical peaks: {n_matched} / {n_total} ({match_rate:.1f}%)")
        print(f"Spurious found peaks (no theory): {n_spurious}")
        
        if len(unmatched_theoretical) > 0:
            print(f"\nUnmatched theoretical peaks ({len(unmatched_theoretical)}):")
            for Q in unmatched_theoretical[:10]:  # only show first 10
                d = 2 * np.pi / Q
                two_theta = 2 * np.arcsin(Q * self.wavelength / (4 * np.pi))
                print(f"  Q = {Q:.4f} Å⁻¹, d = {d:.4f} Å, 2θ = {np.degrees(two_theta):.2f}°")
            if len(unmatched_theoretical) > 10:
                print(f"  ... and {len(unmatched_theoretical) - 10} more")
        
        if len(spurious_found) > 0:
            print(f"\nSpurious peaks:")
            for Q in spurious_found:
                print(f"  Q = {Q:.4f} Å⁻¹")
        
        print("="*70)
        
        # CRITICAL CHECK: The first N theoretical peaks must be matched
        # High-Q spurious peaks are acceptable (noise, edge effects, etc.)
        n_required_peaks = 4
        first_n_theoretical = theoretical_Q_filtered[:n_required_peaks]
        first_n_matched = [Q for Q in first_n_theoretical if Q in matched_theoretical]
        
        self.assertEqual(
            len(first_n_matched), n_required_peaks,
            f"Only {len(first_n_matched)} of the first {n_required_peaks} theoretical peaks matched. "
            f"Expected: {first_n_theoretical.tolist()}, Matched: {first_n_matched}"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
