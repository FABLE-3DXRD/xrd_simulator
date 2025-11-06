"""End-to-end test script for XRD diffraction simulation.

This script demonstrates the complete pipeline:
1. Configure device (GPU/CPU)
2. Load a mesh and create a polycrystal
3. Compute diffraction
4. Render diffraction pattern
5. Save results as TIFF
"""
import os
import time

import numpy as np
import tifffile
import meshio
from scipy.spatial.transform import Rotation as R

# Configure device before importing modules that capture the device at import time
from xrd_simulator import cuda
device = cuda.configure_device(verbose=True)
print(f"Device selected: {device}")

import torch
# Use float64 for nanoscale numerical precision
torch.set_default_dtype(torch.float64)
print(f"Default dtype: {torch.get_default_dtype()}")

from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.beam import Beam
from xrd_simulator.detector import Detector
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.utils import ensure_numpy

import phases


def load_mesh_from_xdmf(filepath):
    """Load a tetrahedral mesh from an XDMF file.
    
    Args:
        filepath (str): Path to the .xdmf mesh file
        
    Returns:
        TetraMesh: Tetrahedral mesh
    """
    mesh = meshio.read(filepath)
    
    # Extract tetrahedral cells only
    tetra_cells = None
    for cell in mesh.cells:
        if cell.type == "tetra":
            tetra_cells = cell.data
            break
    
    if tetra_cells is None:
        raise ValueError("No tetrahedral cells found in mesh")
    
    # Create clean mesh with only tetrahedra
    clean_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("tetra", tetra_cells)]
    )
    
    # Build TetraMesh
    return TetraMesh._build_tetramesh(clean_mesh)


def main():
    """Run the complete diffraction simulation pipeline."""
    dir_this_file = os.path.dirname(os.path.abspath(__file__))
    tiffs_directory = os.path.join("artifacts", "tiffs")

    """Define the parameters for the simulation"""
    beam_params = {
        "energy_kev": 23.0,
        "length_x": 15000000,
        "width_y": 15000000,
        "height_z": 15000000,
    }

    sample_params = {
        "cylinder_radius": 100.0,  # microns
        "cylinder_height": 200.0,  # microns
        "max_cell_circumradius": 30.0,  # microns - controls mesh density
        "phase": "Fe_BCC",
    }

    motion_params = {
        "rotation_axis": np.array([0, 0, 1]),
        "rotation_angle": np.radians(30),
        "translation": np.array([0, 0, 0]),
    }

    detector_dimensions = {
        "samp_to_det_dist": 227_000,  # um
        "pixel_size_z": 172.0,  # um
        "pixel_size_y": 172.0,  # um
        "det_width": 431_000,  # um
        "det_height": 448_000,  # um
    }

    """Define the Beam"""
    beam_vertices = np.array(
        [
            [
                -beam_params["length_x"] * 0.5,
                -beam_params["width_y"] * 0.5,
                -beam_params["height_z"] * 0.5,
            ],
            [
                -beam_params["length_x"] * 0.5,
                beam_params["width_y"] * 0.5,
                -beam_params["height_z"] * 0.5,
            ],
            [
                -beam_params["length_x"] * 0.5,
                beam_params["width_y"] * 0.5,
                beam_params["height_z"] * 0.5,
            ],
            [
                -beam_params["length_x"] * 0.5,
                -beam_params["width_y"] * 0.5,
                beam_params["height_z"] * 0.5,
            ],
            [
                beam_params["length_x"] * 0.5,
                -beam_params["width_y"] * 0.5,
                -beam_params["height_z"] * 0.5,
            ],
            [
                beam_params["length_x"] * 0.5,
                beam_params["width_y"] * 0.5,
                -beam_params["height_z"] * 0.5,
            ],
            [
                beam_params["length_x"] * 0.5,
                beam_params["width_y"] * 0.5,
                beam_params["height_z"] * 0.5,
            ],
            [
                beam_params["length_x"] * 0.5,
                -beam_params["width_y"] * 0.5,
                beam_params["height_z"] * 0.5,
            ],
        ]
    )

    wavelength = 12.398 / beam_params["energy_kev"]

    beam = Beam(
        beam_vertices,
        xray_propagation_direction=np.array([1.0, 0.0, 0.0]),
        wavelength=wavelength,
        polarization_vector=np.array([0.0, 1.0, 0.0]),
    )

    """Define the mesh: Load 10K nanoscale mesh from artifacts"""
    mesh_path = os.path.join(dir_this_file, 'artifacts/samples/nanoscale_10K_R2.5_H5.0.xdmf')
    print(f"Loading nanoscale 10K mesh from: {mesh_path}")
    mesh = load_mesh_from_xdmf(mesh_path)

    print(f"Loaded nanoscale cylindrical mesh with {mesh.number_of_elements} tetrahedral elements")
    print(f"Mesh file: nanoscale_10K_R2.5_H5.0.xdmf (5µm diameter × 5µm height cylinder)")

    """Define the mesh: Random orientations"""
    orientation = R.random(mesh.number_of_elements).as_matrix()

    possible_phases = [phases.a_Ferrite]
    chosen_phases = np.zeros(mesh.number_of_elements)
    chosen_phases[:] = 0
    element_phase_map = list(chosen_phases.astype(int))

    """Define the Sample: Polycrystal"""
    print("\n" + "=" * 80)
    print("Creating Polycrystal")
    print("=" * 80)

    polycrystal = Polycrystal(
        mesh,
        orientation=orientation,
        strain=np.zeros((3, 3)),
        phases=possible_phases,
        element_phase_map=element_phase_map,
    )
    print("✓ Polycrystal created successfully")

    # Save with descriptive filename
    save_filename = f"cylinder_R{sample_params['cylinder_radius']}_H{sample_params['cylinder_height']}_pygmsh.pc"
    polycrystal.save(os.path.join(dir_this_file, 'artifacts/samples', save_filename), save_mesh_as_xdmf=True)
    print(f"Polycrystal saved as: {save_filename}")

    """Define the Motion"""
    motion = RigidBodyMotion(
        rotation_axis=motion_params["rotation_axis"],
        rotation_angle=motion_params["rotation_angle"],
        translation=motion_params["translation"],
    )

    """Define the Detector"""
    detector = Detector(
        pixel_size_z=detector_dimensions["pixel_size_z"],
        pixel_size_y=detector_dimensions["pixel_size_y"],
        det_corner_0=np.array(
            [
                detector_dimensions["samp_to_det_dist"],
                -detector_dimensions["det_width"] * 0.5,
                -detector_dimensions["det_height"] * 0.5,
            ]
        ),
        det_corner_1=np.array(
            [
                detector_dimensions["samp_to_det_dist"],
                detector_dimensions["det_width"] * 0.5,
                -detector_dimensions["det_height"] * 0.5,
            ]
        ),
        det_corner_2=np.array(
            [
                detector_dimensions["samp_to_det_dist"],
                -detector_dimensions["det_width"] * 0.5,
                detector_dimensions["det_height"] * 0.5,
            ]
        ),
    )

    print("\n" + "=" * 80)
    print("Computing Diffraction")
    print("=" * 80)

    start = time.time()

    peaks_dict = polycrystal.diffract(
        beam=beam, detector=detector, rigid_body_motion=motion
    )

    print(f"Diffraction computed in {time.time()-start:.2f} seconds")

    print("\n" + "=" * 80)
    print("Rendering Diffraction Pattern")
    print("=" * 80)

    start_render = time.time()
    diffraction_pattern = detector.render(peaks_dict, frames_to_render=1, method='gauss')

    diffraction_pattern = ensure_numpy(diffraction_pattern)
    diffraction_pattern = diffraction_pattern.astype(np.float32)

    print(f"Rendering completed in {time.time()-start_render:.2f} seconds")
    print(f"Running with {polycrystal.mesh_sample.number_of_elements} tetrahedra")

    file_path = os.path.join(dir_this_file, tiffs_directory, f"run_test.tif")

    tifffile.imwrite(file_path, diffraction_pattern, imagej=True)
    print(f"File saved as {file_path}")


if __name__ == "__main__":
    main()