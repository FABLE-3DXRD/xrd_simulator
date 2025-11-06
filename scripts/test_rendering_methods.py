#!/usr/bin/env python
"""
Comprehensive rendering method comparison test.

This script tests all three rendering methods (volumes, gauss, voigt) on both
CPU and GPU, measures processing times, displays the results side-by-side,
and computes statistical comparisons.

Usage:
    python test_rendering_methods.py
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import phases
import meshio

# Add parent directory to import xrd_simulator
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.beam import Beam
from xrd_simulator.detector import Detector
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.utils import ensure_numpy
from scipy.spatial.transform import Rotation as R


def load_mesh_from_xdmf(filepath):
    """Load a tetrahedral mesh from an XDMF file."""
    mesh = meshio.read(filepath)
    
    tetra_cells = None
    for cell in mesh.cells:
        if cell.type == "tetra":
            tetra_cells = cell.data
            break
    
    if tetra_cells is None:
        raise ValueError("No tetrahedral cells found in mesh")
    
    clean_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("tetra", tetra_cells)]
    )
    
    return TetraMesh._build_tetramesh(clean_mesh)


def setup_simulation():
    """Set up beam, detector, and motion for simulation."""
    
    # Beam parameters
    energy_kev = 23.0
    beam_size = 15000000  # microns
    
    beam_vertices = np.array([
        [-beam_size * 0.5, -beam_size * 0.5, -beam_size * 0.5],
        [-beam_size * 0.5, beam_size * 0.5, -beam_size * 0.5],
        [-beam_size * 0.5, beam_size * 0.5, beam_size * 0.5],
        [-beam_size * 0.5, -beam_size * 0.5, beam_size * 0.5],
        [beam_size * 0.5, -beam_size * 0.5, -beam_size * 0.5],
        [beam_size * 0.5, beam_size * 0.5, -beam_size * 0.5],
        [beam_size * 0.5, beam_size * 0.5, beam_size * 0.5],
        [beam_size * 0.5, -beam_size * 0.5, beam_size * 0.5],
    ])
    
    wavelength = 12.398 / energy_kev
    
    beam = Beam(
        beam_vertices,
        xray_propagation_direction=np.array([1.0, 0.0, 0.0]),
        wavelength=wavelength,
        polarization_vector=np.array([0.0, 1.0, 0.0]),
    )
    
    # Detector parameters
    detector = Detector(
        pixel_size_z=172.0,
        pixel_size_y=172.0,
        det_corner_0=np.array([227_000, -431_000 * 0.5, -448_000 * 0.5]),
        det_corner_1=np.array([227_000, 431_000 * 0.5, -448_000 * 0.5]),
        det_corner_2=np.array([227_000, -431_000 * 0.5, 448_000 * 0.5]),
    )
    
    # Motion
    motion = RigidBodyMotion(
        rotation_axis=np.array([0, 0, 1]),
        rotation_angle=np.radians(30),
        translation=np.array([0, 0, 0]),
    )
    
    return beam, detector, motion


def create_polycrystal(mesh):
    """Create a polycrystal from the mesh."""
    orientation = R.random(mesh.number_of_elements).as_matrix()
    
    possible_phases = [phases.a_Ferrite]
    element_phase_map = [0] * mesh.number_of_elements
    
    polycrystal = Polycrystal(
        mesh,
        orientation=orientation,
        strain=np.zeros((3, 3)),
        phases=possible_phases,
        element_phase_map=element_phase_map,
    )
    
    return polycrystal


def set_device(use_gpu):
    """Set PyTorch device to CPU or GPU and ensure all new tensors use it."""
    if use_gpu and torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    else:
        torch.set_default_device("cpu")
        device = "cpu"
    
    torch.set_default_dtype(torch.float64)
    
    # Clear any cached device state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return device


def render_with_method(detector, peaks_dict, method, device_name):
    """Render diffraction pattern with specified method and time it."""
    print(f"  Rendering with {method} on {device_name}...", end=" ", flush=True)
    
    start = time.time()
    pattern = detector.render(peaks_dict, frames_to_render=1, method=method)
    elapsed = time.time() - start
    
    pattern = ensure_numpy(pattern)
    
    print(f"{elapsed:.3f}s")
    
    return pattern, elapsed


def compute_statistics(pattern):
    """Compute statistics for a diffraction pattern."""
    return {
        'min': pattern.min(),
        'max': pattern.max(),
        'mean': pattern.mean(),
        'std': pattern.std(),
        'sum': pattern.sum(),
        'nonzero': np.count_nonzero(pattern),
    }


def plot_results(results, timings, stats):
    """Create comprehensive visualization of results."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot patterns
    methods = ['volumes', 'gauss', 'voigt']
    devices = ['CPU', 'GPU']

    global_max = max(results[method][device].max() for method in methods for device in devices)
    vmax = 0.9 * global_max  # Set to 90% of max for better contrast

    for i, method in enumerate(methods):
        for j, device in enumerate(devices):
            ax = fig.add_subplot(gs[j, i])
            
            pattern = results[method][device]
            im = ax.imshow(pattern[0], cmap='viridis', vmin=0, vmax=vmax, 
                          interpolation='nearest', origin='lower')
            
            ax.set_title(f"{method.upper()} - {device}\n"
                        f"Time: {timings[method][device]:.3f}s",
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Detector Y (pixels)', fontsize=8)
            ax.set_ylabel('Detector Z (pixels)', fontsize=8)
            ax.tick_params(labelsize=7)
            
    
    # Add timing comparison subplot
    ax_time = fig.add_subplot(gs[2, :2])
    
    x = np.arange(len(methods))
    width = 0.35
    
    cpu_times = [timings[method]['CPU'] for method in methods]
    gpu_times = [timings[method]['GPU'] for method in methods]
    
    bars1 = ax_time.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8, color='steelblue')
    bars2 = ax_time.bar(x + width/2, gpu_times, width, label='GPU', alpha=0.8, color='orangered')
    
    ax_time.set_xlabel('Rendering Method', fontsize=10, fontweight='bold')
    ax_time.set_ylabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax_time.set_title('Rendering Performance Comparison', fontsize=12, fontweight='bold')
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([m.upper() for m in methods])
    ax_time.legend()
    ax_time.grid(axis='y', alpha=0.3)
    
    # Add speedup annotations
    for i, method in enumerate(methods):
        speedup = cpu_times[i] / gpu_times[i] if gpu_times[i] > 0 else 0
        ax_time.text(i, max(cpu_times[i], gpu_times[i]) * 1.05, 
                    f'{speedup:.1f}x', ha='center', fontsize=8, fontweight='bold')
    
    # Add statistics comparison
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    
    stats_text = "Pattern Statistics Comparison:\n\n"
    for method in methods:
        stats_text += f"{method.upper()}:\n"
        cpu_s = stats[method]['CPU']
        gpu_s = stats[method]['GPU']
        
        # Compare CPU and GPU results
        diff = abs(cpu_s['sum'] - gpu_s['sum']) / cpu_s['sum'] * 100 if cpu_s['sum'] > 0 else 0
        
        stats_text += f"  Sum: CPU={cpu_s['sum']:.2e}, GPU={gpu_s['sum']:.2e}\n"
        stats_text += f"  Difference: {diff:.4f}%\n"
        stats_text += f"  Max: CPU={cpu_s['max']:.2e}, GPU={gpu_s['max']:.2e}\n"
        stats_text += f"  Nonzero: CPU={cpu_s['nonzero']}, GPU={gpu_s['nonzero']}\n\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('XRD Simulator: Rendering Method Comparison (CPU vs GPU)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    return fig


def main():
    """Main test function."""
    print("=" * 80)
    print("XRD Simulator Rendering Method Comparison Test")
    print("=" * 80)
    print()
    
    # Load mesh
    mesh_path = os.path.join(script_dir, 'artifacts/samples/macroscopic_10_R500_H1000.xdmf')
    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh_from_xdmf(mesh_path)
    print(f"✓ Loaded mesh with {mesh.number_of_elements} elements\n")
    
    # Test parameters
    methods = ['volumes', 'gauss', 'voigt']
    devices_config = [
        ('CPU', False),
        ('GPU', True),
    ]
    
    # Storage for results
    results = {method: {} for method in methods}
    timings = {method: {} for method in methods}
    stats = {method: {} for method in methods}
    
    # Run tests for each device
    for device_name, use_gpu in devices_config:
        if use_gpu and not torch.cuda.is_available():
            print(f"⚠ GPU requested but CUDA not available, skipping GPU tests")
            continue
        
        print("=" * 80)
        print(f"Testing on {device_name}")
        print("=" * 80)
        
        # Set device
        device = set_device(use_gpu)
        print(f"Device: {device}")
        print(f"Default dtype: {torch.get_default_dtype()}\n")
        
        # Reload mesh on the correct device (important for device consistency)
        print("Loading mesh on current device...")
        mesh = load_mesh_from_xdmf(mesh_path)
        print(f"✓ Mesh loaded with {mesh.number_of_elements} elements\n")
        
        # Set up simulation objects (must be created AFTER device is set)
        print("Setting up simulation objects...")
        beam, detector, motion = setup_simulation()
        print("✓ Beam, detector, and motion configured\n")
        
        # Create polycrystal (must be created on the correct device)
        print("Creating polycrystal...")
        polycrystal = create_polycrystal(mesh)
        print(f"✓ Polycrystal created with {polycrystal.mesh_sample.number_of_elements} elements\n")
        
        # Compute diffraction (once per device)
        print("Computing diffraction with convex hulls...")
        start = time.time()
        peaks_dict = polycrystal.diffract(
            beam=beam, 
            detector=detector, 
            rigid_body_motion=motion)
        diffraction_time = time.time() - start
        print(f"✓ Diffraction computed in {diffraction_time:.3f}s")
        print(f"  Number of peaks: {len(peaks_dict['peaks'])}")
        if 'convex_hulls' in peaks_dict:
            print(f"  Convex hulls generated: {len(peaks_dict['convex_hulls'])}")
        print()
        
        # Render with each method
        print("Rendering patterns:")
        for method in methods:
            try:
                pattern, elapsed = render_with_method(detector, peaks_dict, method, device_name)
                results[method][device_name] = pattern
                timings[method][device_name] = elapsed
                stats[method][device_name] = compute_statistics(pattern)
            except KeyError as e:
                if 'convex_hulls' in str(e):
                    print(f"  ⚠ {method} requires convex_hulls (not available), skipping")
                    # Create dummy pattern matching detector shape
                    shape = detector.pixel_coordinates.shape
                    dummy_pattern = np.zeros((1, shape[0], shape[1]))
                    results[method][device_name] = dummy_pattern
                    timings[method][device_name] = 0
                    stats[method][device_name] = compute_statistics(dummy_pattern)
                else:
                    raise
            except Exception as e:
                print(f"  ✗ Error rendering with {method}: {e}")
                shape = detector.pixel_coordinates.shape
                dummy_pattern = np.zeros((1, shape[0], shape[1]))
                results[method][device_name] = dummy_pattern
                timings[method][device_name] = 0
                stats[method][device_name] = compute_statistics(dummy_pattern)
        
        print()
    
    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    print("Timing Results:")
    print("-" * 80)
    print(f"{'Method':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
    print("-" * 80)
    
    for method in methods:
        cpu_time = timings[method].get('CPU', 0)
        gpu_time = timings[method].get('GPU', 0)
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"{method.upper():<12} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<12.2f}x")
    
    print()
    
    # Verify numerical agreement
    print("Numerical Agreement:")
    print("-" * 80)
    
    for method in methods:
        if stats[method].get('CPU') and stats[method].get('GPU'):
            cpu_sum = stats[method]['CPU']['sum']
            gpu_sum = stats[method]['GPU']['sum']
            diff = abs(cpu_sum - gpu_sum) / cpu_sum * 100 if cpu_sum > 0 else 0
            
            status = "✓ MATCH" if diff < 0.1 else "⚠ DIFFER"
            print(f"{method.upper():<12} CPU sum: {cpu_sum:.6e}  GPU sum: {gpu_sum:.6e}  "
                  f"Diff: {diff:.4f}%  {status}")
    
    print()
    
    # Create visualization
    print("Creating visualization...")
    fig = plot_results(results, timings, stats)
    
    # Save figure
    output_path = os.path.join(script_dir, 'artifacts/tiffs/rendering_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Show figure
    print("\nDisplaying results...")
    plt.show()
    
    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
