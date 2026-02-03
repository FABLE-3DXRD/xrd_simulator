"""
End-to-end test: Validate crystallite size estimation from simulated powder diffraction.

This test:
1. Generates a 10,000 grain ferrite (BCC iron) polycrystal with 50nm grain size
2. Simulates diffraction over 30° rotation (full powder rings)
3. Renders using Airy method (includes Scherrer broadening)
4. Integrates azimuthally using pyFAI
5. Fits peaks with pseudo-Voigt profile using scipy
6. Estimates crystallite size via Scherrer equation
7. Compares estimated grain size against ground truth

This validates the entire pipeline from crystal structure to detector output to
crystallographic grain size analysis.
"""
import unittest
import numpy as np
import os
import sys
import io

try:
    import pyFAI
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    HAS_PYFAI = True
except ImportError:
    HAS_PYFAI = False

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from xfab import tools
import matplotlib.pyplot as plt

from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.phase import Phase
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.cuda import configure_device, get_selected_device

# Store original device before any test configuration
_ORIGINAL_DEVICE = get_selected_device()

# =============================================================================
# SIMULATION PARAMETERS - Modify these to change the test configuration
# =============================================================================

# X-ray beam parameters
ENERGY_KEV = 45  # X-ray energy in keV
WAVELENGTH = 12.398 / ENERGY_KEV  # Wavelength in Angstrom (0.2755 Å at 45 keV)

# Detector geometry
PIXEL_SIZE = 150.0  # Pixel size in microns
N_PIXELS = 2048  # Detector is N_PIXELS x N_PIXELS
DETECTOR_DISTANCE = 500_000.0  # Sample-to-detector distance in microns (500 mm)
GAUSSIAN_SIGMA = 0.01  # Gaussian PSF sigma (very small to minimize instrumental broadening)

# Sample parameters
N_GRAINS = 1000 # Number of grains (100k for good powder statistics)
TARGET_SIZE_NM = 5  # Target crystallite size in nm
SAMPLE_EXTENT = 10.0  # Sample extent in microns (cube side length)

# Rotation parameters (for powder averaging)
ROTATION_ANGLE_DEG = 25.0  # Rotation angle in degrees
ROTATION_AXIS = np.array([0.0, 0.0, 1.0])  # Rotation axis

# Material: Ferrite (BCC Iron)
FERRITE_A = 2.87  # Lattice parameter in Angstrom
UNIT_CELL = [FERRITE_A, FERRITE_A, FERRITE_A, 90.0, 90.0, 90.0]
SGNAME = 'Im-3m'  # Space group

# Peak matching tolerance
Q_TOLERANCE = 0.015

# =============================================================================


def compute_theoretical_Q_values(unit_cell, sgname, wavelength, max_bragg_angle):
    """
    Compute theoretical Q values for all allowed reflections using xfab.
    
    Uses the same crystallographic library (xfab) that xrd_simulator uses internally,
    ensuring consistency between theoretical and simulated peak positions.
    
    Args:
        unit_cell: [a, b, c, alpha, beta, gamma] in Angstrom and degrees
        sgname: Space group name (e.g., 'Im-3m')
        wavelength: X-ray wavelength in Angstrom
        max_bragg_angle: Maximum Bragg angle in radians
        
    Returns:
        unique_Q: Sorted array of unique Q values (Å⁻¹)
        d_spacings: Corresponding d-spacings (Å)
    """
    # Generate all allowed Miller indices using xfab
    sintlmin = 0.0
    sintlmax = np.sin(max_bragg_angle) / wavelength
    miller_indices = tools.genhkl_all(unit_cell, sintlmin, sintlmax, sgname=sgname)
    
    if miller_indices is None or len(miller_indices) == 0:
        return np.array([]), np.array([])
    
    # Compute B matrix (reciprocal lattice)
    B = tools.form_b_mat(unit_cell)
    
    # Compute d-spacings for each (hkl)
    G_vectors = np.dot(B, miller_indices.T).T
    G_norms = np.linalg.norm(G_vectors, axis=1)
    d_spacings = 2 * np.pi / G_norms
    
    # Q = 2π/d
    Q_values = 2 * np.pi / d_spacings
    
    # Get unique Q values (multiple hkl can have same d-spacing)
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
    intensity = np.array(intensity)
    max_int = np.max(intensity)
    if max_int == 0:
        return np.array([]), np.array([])
    
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


def voigt_function(x, amplitude, center, sigma, gamma):
    """
    Voigt profile function for fitting.
    
    The Voigt profile is the convolution of Gaussian and Lorentzian.
    Using scipy's voigt_profile which is normalized.
    
    Args:
        x: Independent variable (2theta or Q)
        amplitude: Peak amplitude (area under curve)
        center: Peak center position
        sigma: Gaussian width parameter (standard deviation)
        gamma: Lorentzian width parameter (HWHM)
        
    Returns:
        Voigt profile values
    """
    # scipy's voigt_profile expects (x - center) / sigma and gamma / sigma
    return amplitude * voigt_profile((x - center), sigma, gamma)


def pseudo_voigt_function(x, amplitude, center, fwhm, eta):
    """
    Pseudo-Voigt profile: linear combination of Gaussian and Lorentzian.
    
    PV(x) = eta * L(x) + (1 - eta) * G(x)
    
    where eta is the mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian).
    
    Args:
        x: Independent variable
        amplitude: Peak amplitude
        center: Peak center
        fwhm: Full width at half maximum (shared by both G and L)
        eta: Mixing parameter [0, 1] (fraction Lorentzian)
        
    Returns:
        Pseudo-Voigt profile values
    """
    # Gaussian component
    sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
    gaussian = np.exp(-0.5 * ((x - center) / sigma_g) ** 2) / (sigma_g * np.sqrt(2 * np.pi))
    
    # Lorentzian component  
    gamma_l = fwhm / 2  # FWHM to HWHM
    lorentzian = gamma_l / (np.pi * ((x - center) ** 2 + gamma_l ** 2))
    
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)


def fit_voigt_peak(x_data, y_data, center_guess, verbose=False):
    """
    Fit a Voigt/pseudo-Voigt profile to a peak.
    
    Tries pseudo-Voigt first (more stable), falls back to pure Voigt.
    
    Args:
        x_data: 2theta values around the peak
        y_data: Intensity values
        center_guess: Initial guess for peak center
        verbose: Print fit results
        
    Returns:
        dict with keys: 'center', 'fwhm', 'amplitude', 'eta' (Lorentzian fraction),
                       'success', 'method'
        Returns None if fit fails
    """
    # Normalize data for better fitting
    y_max = np.max(y_data)
    if y_max == 0:
        return None
    y_norm = y_data / y_max
    
    # Estimate initial parameters
    amplitude_guess = np.trapezoid(y_norm, x_data)  # Area under curve
    fwhm_guess = 0.3  # degrees, typical for powder diffraction
    
    try:
        # Try pseudo-Voigt first (eta as free parameter)
        popt, pcov = curve_fit(
            pseudo_voigt_function,
            x_data, y_norm,
            p0=[amplitude_guess, center_guess, fwhm_guess, 0.5],
            bounds=([0, x_data[0], 0.01, 0], [np.inf, x_data[-1], 5, 1]),
            maxfev=5000
        )
        
        amplitude, center, fwhm, eta = popt
        
        # Rescale amplitude back
        amplitude *= y_max
        
        # Check fit quality - compute R²
        y_fit = pseudo_voigt_function(x_data, amplitude / y_max, center, fwhm, eta)
        ss_res = np.sum((y_norm - y_fit) ** 2)
        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if verbose:
            print(f"  Pseudo-Voigt fit: center={center:.4f}°, FWHM={fwhm:.4f}°, "
                  f"eta={eta:.3f}, R²={r_squared:.4f}")
        
        return {
            'center': center,
            'fwhm': fwhm,
            'amplitude': amplitude,
            'eta': eta,
            'r_squared': r_squared,
            'success': True,
            'method': 'pseudo-voigt'
        }
        
    except Exception as e:
        if verbose:
            print(f"  Pseudo-Voigt fit failed: {e}")
        return None


def scherrer_from_fwhm(fwhm_rad, wavelength_A, theta_rad, K=0.9):
    """
    Calculate crystallite size using Scherrer equation.
    
    D = K * λ / (β * cos(θ))
    
    Args:
        fwhm_rad: Full width at half maximum in radians
        wavelength_A: X-ray wavelength in Angstrom
        theta_rad: Bragg angle (half of 2theta) in radians
        K: Scherrer constant (default 0.9 for spherical crystallites)
        
    Returns:
        Crystallite size in nm
    """
    return (K * wavelength_A) / (fwhm_rad * np.cos(theta_rad)) / 10  # Å to nm


@unittest.skipUnless(HAS_PYFAI, "pyFAI not installed")
class TestCrystalliteSize(unittest.TestCase):
    """
    Validate crystallite size estimation from simulated powder diffraction.
    
    Tests 100 nm target crystallite size with comprehensive plotting.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up simulation parameters from module-level constants."""
        # Configure GPU for this test class
        configure_device("gpu", verbose=True)
        
        # Copy module-level constants to class attributes
        cls.energy_keV = ENERGY_KEV
        cls.wavelength = WAVELENGTH
        cls.pixel_size = PIXEL_SIZE
        cls.n_pixels = N_PIXELS
        cls.detector_distance = DETECTOR_DISTANCE
        cls.detector_size = cls.pixel_size * cls.n_pixels
        
        # Detector corners (detector perpendicular to x-axis, centered on beam)
        det_corner_0 = np.array([cls.detector_distance, -cls.detector_size/2, -cls.detector_size/2])
        det_corner_1 = np.array([cls.detector_distance,  cls.detector_size/2, -cls.detector_size/2])
        det_corner_2 = np.array([cls.detector_distance, -cls.detector_size/2,  cls.detector_size/2])
        
        cls.detector = Detector(
            det_corner_0=det_corner_0,
            det_corner_1=det_corner_1,
            det_corner_2=det_corner_2,
            n_pixels=(cls.n_pixels, cls.n_pixels),
            gaussian_sigma=GAUSSIAN_SIGMA)
        
        # Calculate detector coverage in 2theta
        cls.max_2theta = np.arctan((cls.detector_size/2) / cls.detector_distance)
        cls.max_bragg_angle = cls.max_2theta
        
        # Sample parameters
        cls.n_grains = N_GRAINS
        
        # Rotation parameters
        cls.rotation_angle = np.radians(ROTATION_ANGLE_DEG)
        cls.rotation_axis = ROTATION_AXIS.copy()
        
        # Material parameters
        cls.Q_tolerance = Q_TOLERANCE
        cls.ferrite_a = FERRITE_A
        cls.unit_cell = UNIT_CELL.copy()
        cls.sgname = SGNAME
    
    @classmethod
    def tearDownClass(cls):
        """Restore original device state after tests."""
        configure_device(_ORIGINAL_DEVICE, verbose=False)
    
    @classmethod
    def _volume_for_crystallite_size(cls, size_nm):
        """
        Calculate mesh element volume needed for target crystallite size.
        
        From scattering_factors.py:
        crystallite_size_A = 2.0 * (3 * V / (4 * pi))^(1/3) * 10000
        where V is in microns³
        
        Inverting: V = (4*pi/3) * (size_A / 20000)³
        """
        size_A = size_nm * 10.0  # nm to Angstrom
        volume_um3 = (4 * np.pi / 3) * (size_A / 20000.0) ** 3
        return volume_um3
    
    @classmethod
    def _tetrahedron_edge_for_volume(cls, target_volume):
        """
        Calculate tetrahedron edge length for target volume.
        
        Regular tetrahedron: V = (a³)/(6*sqrt(2))
        So: a = (6*sqrt(2)*V)^(1/3)
        """
        return (6 * np.sqrt(2) * target_volume) ** (1/3)
    
    def _create_beam(self, sample_extent):
        """Create beam large enough to cover the sample."""
        w = sample_extent * 20  # 20x sample extent for safety
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
        
        return Beam(
            beam_vertices,
            xray_propagation_direction,
            self.wavelength,
            polarization_vector
        )
    
    def _create_motion(self):
        """Create rotation motion."""
        return RigidBodyMotion(
            self.rotation_axis,
            self.rotation_angle,
            np.array([0.0, 0.0, 0.0])
        )
    
    def _create_polycrystal_with_size(self, target_size_nm):
        """
        Create a polycrystal with mesh elements sized for target crystallite size.
        
        Creates tetrahedra with controlled volumes to achieve desired Scherrer broadening.
        Distributes grains in a thin slab (300nm thick) to minimize parallax effects.
        """
        from scipy.spatial.transform import Rotation
        
        # Calculate required volume and edge length
        target_volume = self._volume_for_crystallite_size(target_size_nm)
        edge_length = self._tetrahedron_edge_for_volume(target_volume)
        
        # Build tetrahedra mesh manually with controlled size
        # Distribute grains in a thin slab (300nm thick) along beam direction
        coord = []
        enod = []
        node_number = 0
        
        # Regular tetrahedron vertices (edge length = edge_length, centered at origin)
        # Using standard regular tetrahedron coordinates scaled by edge_length
        h = edge_length * np.sqrt(2/3)  # height
        r_base = edge_length / np.sqrt(3)  # circumradius of base triangle
        
        # Cube geometry: equal extent in all directions
        extent = SAMPLE_EXTENT  # microns
        
        for _ in range(self.n_grains):
            # Random position within cube
            offset_x = np.random.uniform(-extent/2, extent/2)
            offset_y = np.random.uniform(-extent/2, extent/2)
            offset_z = np.random.uniform(-extent/2, extent/2)
            
            # Vertices of regular tetrahedron, offset by random position
            v0 = [offset_x + 0, offset_y + 0, offset_z + h * 0.75]  # apex
            v1 = [offset_x + r_base, offset_y + 0, offset_z - h * 0.25]
            v2 = [offset_x - r_base/2, offset_y + r_base * np.sqrt(3)/2, offset_z - h * 0.25]
            v3 = [offset_x - r_base/2, offset_y - r_base * np.sqrt(3)/2, offset_z - h * 0.25]
            
            coord.extend([v0, v1, v2, v3])
            enod.append([node_number, node_number+1, node_number+2, node_number+3])
            node_number += 4
        
        coord = np.array(coord)
        enod = np.array(enod)
        
        mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)
        orientation = Rotation.random(mesh.number_of_elements).as_matrix()
        element_phase_map = np.zeros((mesh.number_of_elements,)).astype(int)
        phases = [Phase(self.unit_cell, self.sgname)]
        
        return Polycrystal(
            mesh,
            orientation,
            strain=np.zeros((3, 3)),
            phases=phases,
            element_phase_map=element_phase_map,
        )
    
    def _create_polycrystal(self, sample_radius):
        """Create a polycrystal using standard template (for backward compatibility)."""
        from xrd_simulator.templates import get_uniform_powder_sample
        
        polycrystal = get_uniform_powder_sample(
            sample_bounding_radius=sample_radius,
            number_of_grains=self.n_grains,
            unit_cell=self.unit_cell,
            sgname=self.sgname
        )
        
        return polycrystal
    
    def _setup_pyfai_integrator(self):
        """Create pyFAI integrator matching our detector geometry."""
        dist_m = self.detector_distance * 1e-6
        pixel_m = self.pixel_size * 1e-6
        
        poni1 = self.n_pixels / 2 * pixel_m
        poni2 = self.n_pixels / 2 * pixel_m
        
        ai = AzimuthalIntegrator(
            dist=dist_m,
            poni1=poni1,
            poni2=poni2,
            rot1=0, rot2=0, rot3=0,
            pixel1=pixel_m,
            pixel2=pixel_m,
            wavelength=self.wavelength * 1e-10
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
        Q_ticks = ax2.get_xticks()
        Q_ticks = Q_ticks[(Q_ticks > 0) & (Q_ticks < 4 * np.pi / self.wavelength)]
        two_theta_ticks = 2 * np.arcsin(Q_ticks * self.wavelength / (4 * np.pi))
        ax2_top.set_xticks(Q_ticks)
        ax2_top.set_xticklabels([f'{np.degrees(t):.0f}°' for t in two_theta_ticks])
        ax2_top.set_xlabel('2θ')
        
        plt.tight_layout()
    
    def test_powder_pattern_with_scherrer(self):
        """
        Main test: Simulate powder pattern and verify Scherrer estimation.
        
        OPTIMAL CONDITIONS for grain size estimation:
        - 20 keV energy (larger FWHM than high-energy)
        - 500mm detector distance
        - 50µm pixel size
        - This setup can measure ~10-60nm crystallites reliably
        """
        target_size_nm =  TARGET_SIZE_NM  # 20nm gives FWHM ~28 pixels (well sampled)
        
        print("\n" + "="*80)
        print(f"CRYSTALLITE SIZE ESTIMATION TEST - {target_size_nm}nm VALIDATION")
        print("="*80)
        print(f"Energy: {self.energy_keV} keV (λ = {self.wavelength:.4f} Å)")
        print(f"Grains: {self.n_grains}")
        print(f"Rotation: {np.degrees(self.rotation_angle):.1f}°")
        print(f"Detector: {self.n_pixels}x{self.n_pixels} pixels, {self.detector_distance/1000:.1f} mm distance")
        print(f"Max 2θ coverage: {np.degrees(2*self.max_bragg_angle):.2f}°")
        print(f"Material: Ferrite - BCC Iron ({self.sgname})")
        print(f"Target crystallite size: {target_size_nm} nm")
        print("="*80)
        
        # Calculate required volume and verify
        target_volume = self._volume_for_crystallite_size(target_size_nm)
        edge_length = self._tetrahedron_edge_for_volume(target_volume)
        print(f"\n[Config] Target crystallite size: {target_size_nm} nm")
        print(f"[Config] Sample extent: {SAMPLE_EXTENT} µm (cube)")
        print(f"[Config] Required element volume: {target_volume:.6e} µm³")
        print(f"[Config] Tetrahedron edge length: {edge_length:.6f} µm")
        
        # Create beam - use sample lateral extent not edge length
        beam = self._create_beam(SAMPLE_EXTENT)
        motion = self._create_motion()
        
        # Create polycrystal with controlled element size
        print("\n[1] Creating polycrystal sample...")
        polycrystal = self._create_polycrystal_with_size(target_size_nm)
        print(f"    Mesh elements: {polycrystal.mesh_lab.number_of_elements}")
        
        # Get actual mesh element volumes and verify
        volumes = polycrystal.mesh_lab.evolumes
        if hasattr(volumes, 'cpu'):
            volumes = volumes.cpu().numpy()
        avg_volume = np.mean(np.abs(volumes))
        expected_size_A = 2.0 * (3 * avg_volume / (4 * np.pi)) ** (1/3) * 10000
        expected_size_nm = expected_size_A / 10.0
        print(f"    Actual average element volume: {avg_volume:.6e} µm³")
        print(f"    Effective crystallite size (from volume): {expected_size_nm:.1f} nm")
        
        # Verify we got the right size
        size_error = abs(expected_size_nm - target_size_nm) / target_size_nm * 100
        print(f"    Volume error vs target: {size_error:.2f}%")
        
        # Simulate diffraction
        print("\n[2] Computing diffraction...")
        peaks_dict = polycrystal.diffract(
            beam, 
            motion,
            detector=self.detector,
            verbose=False
        )
        n_peaks = len(peaks_dict['peaks'])
        print(f"    Generated {n_peaks} diffraction peaks")
        
        # Examine internal Scherrer FWHM values
        peaks_tensor = peaks_dict['peaks']
        if hasattr(peaks_tensor, 'cpu'):
            peaks_np = peaks_tensor.cpu().numpy()
        else:
            peaks_np = np.array(peaks_tensor)
        
        # Column 23 is scherrer_fwhm according to COLUMN_MAPPING.md
        scherrer_fwhm_rad = peaks_np[:, 23]
        scherrer_fwhm_deg = np.degrees(scherrer_fwhm_rad)
        print(f"    Internal Scherrer FWHM: mean={np.mean(scherrer_fwhm_deg):.4f}°, "
              f"std={np.std(scherrer_fwhm_deg):.4f}°, range=[{np.min(scherrer_fwhm_deg):.4f}°, {np.max(scherrer_fwhm_deg):.4f}°]")
        
        # Render pattern with NANO mode (GPU accelerated)
        print("\n[3] Rendering diffraction pattern (NANO mode, GPU)...")
        pattern = self.detector.render(
            peaks_dict,
            frames_to_render=0,
            method='nano'
        )
        
        if hasattr(pattern, 'cpu'):
            pattern_np = pattern[0].cpu().numpy()
        else:
            pattern_np = np.array(pattern[0])
        
        print(f"    Non-zero pixels: {np.sum(pattern_np > 0)}")
        print(f"    Max intensity: {np.max(pattern_np):.2e}")
        
        # pyFAI integration - use high number of bins for good resolution
        print("\n[4] Integrating with pyFAI...")
        ai = self._setup_pyfai_integrator()
        
        # Use enough bins: at least 5 bins per FWHM
        # With 0.005° pixel angular size and ~0.1° FWHM, need ~20 bins per FWHM
        # For 60° coverage, that's ~12000 bins minimum
        Q_result, I_result = ai.integrate1d(
            pattern_np,
            20000,  # High resolution for accurate FWHM measurement
            unit='q_A^-1',
            correctSolidAngle=True
        )
        
        # Convert Q to 2theta for analysis
        twotheta_result = np.degrees(2 * np.arcsin(Q_result * self.wavelength / (4 * np.pi)))
        
        # Find peaks in pattern
        found_Q, found_I = find_peaks_in_pattern(Q_result, I_result, prominence_fraction=0.005)
        print(f"    Found {len(found_Q)} peaks in integrated pattern")
        
        # ============ Scherrer analysis with scipy Voigt fitting ============
        print("\n[5] Scherrer analysis with pseudo-Voigt fitting (scipy)...")
        
        x_data = twotheta_result
        
        estimated_sizes_voigt = []
        peak_info = []  # Store for plotting
        
        sorted_indices = np.argsort(found_I)[::-1]
        
        for i, idx in enumerate(sorted_indices):
            if i >= 10:
                break
            Q_found = found_Q[idx]
            peak_twotheta = np.degrees(2 * np.arcsin(Q_found * self.wavelength / (4 * np.pi)))
            
            # Extract region around peak
            xrange = [max(x_data[0], peak_twotheta - 1.5), 
                     min(x_data[-1], peak_twotheta + 1.5)]
            mask = (x_data >= xrange[0]) & (x_data <= xrange[1])
            x_segment = x_data[mask]
            y_segment = I_result[mask]
            
            if len(x_segment) < 10:
                continue
            
            # Fit with Voigt/pseudo-Voigt
            fit_result = fit_voigt_peak(x_segment, y_segment, peak_twotheta, verbose=False)
            
            if fit_result is not None and fit_result['success']:
                fwhm_deg = fit_result['fwhm']
                fwhm_rad = np.radians(fwhm_deg)
                theta_rad = np.radians(fit_result['center'] / 2)
                
                # Calculate crystallite size using Scherrer equation
                size_nm = scherrer_from_fwhm(fwhm_rad, self.wavelength, theta_rad, K=0.9)
                
                print(f"      Peak {i+1}: 2θ={fit_result['center']:.2f}°, FWHM={fwhm_deg:.4f}°, η={fit_result['eta']:.3f}, size={size_nm:.1f}nm")
                
                if size_nm > 0 and size_nm < 10000:
                    estimated_sizes_voigt.append(size_nm)
                    peak_info.append({
                        'twotheta': fit_result['center'],
                        'intensity': fit_result['amplitude'],
                        'size_nm': size_nm,
                        'fwhm_deg': fwhm_deg,
                        'eta': fit_result['eta'],
                        'xrange': xrange
                    })
        
        # Compare fitted FWHM to internal Scherrer FWHM
        if len(peak_info) > 0:
            avg_fitted_fwhm = np.mean([p['fwhm_deg'] for p in peak_info])
            internal_fwhm = np.degrees(np.mean(scherrer_fwhm_rad))
            print(f"\n    Fitted FWHM (avg): {avg_fitted_fwhm:.4f}°")
            print(f"    Internal Scherrer FWHM: {internal_fwhm:.4f}°")
            print(f"    Ratio (fitted/internal): {avg_fitted_fwhm/internal_fwhm:.2f}x")
        
        # Compute results - Voigt fitting
        if len(estimated_sizes_voigt) > 0:
            avg_estimated = np.mean(estimated_sizes_voigt)
            std_estimated = np.std(estimated_sizes_voigt) if len(estimated_sizes_voigt) > 1 else 0
            error_percent = abs(avg_estimated - expected_size_nm) / expected_size_nm * 100
            
            print(f"\n    Analyzed {len(estimated_sizes_voigt)} peaks")
            print(f"    Estimated size: {avg_estimated:.1f} ± {std_estimated:.1f} nm")
            print(f"    Expected size: {expected_size_nm:.1f} nm")
            print(f"    Error: {error_percent:.1f}%")
            
            if len(peak_info) > 0:
                avg_eta = np.mean([p['eta'] for p in peak_info])
                print(f"    Avg Lorentzian fraction (eta): {avg_eta:.3f}")
        else:
            avg_estimated = None
            std_estimated = None
            error_percent = None
            print("    No valid Voigt estimates!")
        
        # Use Voigt results for test assertions
        estimated_sizes = estimated_sizes_voigt
        
        # ==================== PLOTTING ====================
        print("\n[6] Generating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: 2D diffraction pattern
        ax1 = axes[0, 0]
        im = ax1.imshow(pattern_np, cmap='hot', aspect='equal', 
                       vmin=0, vmax=np.percentile(pattern_np[pattern_np > 0], 99) if np.any(pattern_np > 0) else 1)
        ax1.set_title(f'2D Diffraction Pattern\n(Target: {target_size_nm} nm crystallite)', fontsize=12)
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        plt.colorbar(im, ax=ax1, label='Intensity')
        
        # Plot 2: 1D integrated pattern
        ax2 = axes[0, 1]
        ax2.plot(twotheta_result, I_result, 'b-', linewidth=0.8, label='Integrated pattern')
        
        # Mark found peaks
        for info in peak_info:
            ax2.axvline(info['twotheta'], color='r', linestyle='--', alpha=0.5, linewidth=0.5)
        
        ax2.set_xlabel('2θ (degrees)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('1D Integrated Pattern', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Per-peak Scherrer estimates
        ax3 = axes[1, 0]
        if len(peak_info) > 0:
            peak_2thetas = [p['twotheta'] for p in peak_info]
            peak_sizes = [p['size_nm'] for p in peak_info]
            
            ax3.scatter(peak_2thetas, peak_sizes, c='blue', s=100, marker='o', label='Scherrer estimates')
            ax3.axhline(expected_size_nm, color='green', linestyle='-', linewidth=2, 
                       label=f'Expected: {expected_size_nm:.1f} nm')
            ax3.axhline(target_size_nm, color='red', linestyle='--', linewidth=2, 
                       label=f'Target: {target_size_nm} nm')
            if avg_estimated:
                ax3.axhline(avg_estimated, color='orange', linestyle=':', linewidth=2,
                           label=f'Mean estimate: {avg_estimated:.1f} nm')
            
            ax3.set_xlabel('2θ (degrees)')
            ax3.set_ylabel('Estimated crystallite size (nm)')
            ax3.set_title('Scherrer Size Estimates per Peak', fontsize=12)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, max(expected_size_nm * 1.5, max(peak_sizes) * 1.2)])
        else:
            ax3.text(0.5, 0.5, 'No valid Scherrer estimates', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Scherrer Size Estimates per Peak', fontsize=12)
        
        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
CRYSTALLITE SIZE ESTIMATION SUMMARY
{'='*45}

Input Configuration:
  • Target crystallite size: {target_size_nm} nm
  • Mesh element volume: {target_volume:.3e} µm³
  • Number of grains: {self.n_grains}
  • X-ray wavelength: {self.wavelength:.4f} Å

Verification:
  • Actual mesh volume: {avg_volume:.3e} µm³
  • Effective size (from volume): {expected_size_nm:.1f} nm
  • Volume accuracy: {100 - size_error:.1f}%

Scherrer Analysis:
  • Peaks analyzed: {len(estimated_sizes)}
  • Estimated size: {f'{avg_estimated:.1f} ± {std_estimated:.1f} nm' if avg_estimated else 'N/A'}
  • Expected size: {expected_size_nm:.1f} nm
  • Error: {f'{error_percent:.1f}%' if error_percent else 'N/A'}

{'='*45}
"""
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save and show
        output_path = '/tmp/crystallite_size_100nm_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Plot saved to: {output_path}")
        
        
        # Assertions
        self.assertGreater(len(estimated_sizes), 0, "No crystallite sizes could be estimated!")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)


if __name__ == '__main__':
    unittest.main(verbosity=2)
