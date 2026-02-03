import matplotlib.pyplot as plt
import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xfab import tools
import os
import cProfile
import pstats

np.random.seed(23)

grainmeshfile = os.path.join(
    os.path.join(
        os.path.dirname(__file__),
        '../../../tests/data'),
    'grain0056.xdmf')
mesh = TetraMesh.load(grainmeshfile)

sample_diameter = 1.0

print("")
print('nelm:', mesh.number_of_elements)
print("")

# ============================================================
# STEP 1: Wide detector to find where peaks land
# ============================================================
pixel_size = sample_diameter / 256.
detector_size = pixel_size * 2048  # 10% larger to capture edge peaks
detector_distance = 10 * sample_diameter

# Wide detector centered on direct beam
det_corner_0 = np.array([detector_distance, -detector_size / 2., -detector_size / 2.])
det_corner_1 = np.array([detector_distance, detector_size / 2., -detector_size / 2.])
det_corner_2 = np.array([detector_distance, -detector_size / 2., detector_size / 2.])

detector_wide = Detector(
    det_corner_0=det_corner_0,
    det_corner_1=det_corner_1,
    det_corner_2=det_corner_2,
    pixel_size=pixel_size)

unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
sgname = 'Fm-3m'  # Iron
phases = [Phase(unit_cell, sgname)]

grain_avg_rot = np.max([np.radians(1.0), np.random.rand() * 2 * np.pi])
euler_angles = grain_avg_rot + \
    np.random.normal(loc=0.0, scale=np.radians(0.01), size=(mesh.number_of_elements, 3))
orientation = np.array([tools.euler_to_u(ea[0], ea[1], ea[2])
                       for ea in euler_angles])
element_phase_map = np.zeros((mesh.number_of_elements,)).astype(int)
polycrystal = Polycrystal(mesh, orientation, strain=np.zeros(
    (3, 3)), phases=phases, element_phase_map=element_phase_map)

w = detector_size  # full field beam
beam_vertices = np.array([
    [-detector_distance, -w, -w],
    [-detector_distance, w, -w],
    [-detector_distance, w, w],
    [-detector_distance, -w, w],
    [detector_distance, -w, -w],
    [detector_distance, w, -w],
    [detector_distance, w, w],
    [detector_distance, -w, w]])
wavelength = 0.285227
xray_propagation_direction = np.array([1, 0, 0]) * 2 * np.pi / wavelength
polarization_vector = np.array([0, 1, 0])
beam = Beam(
    beam_vertices,
    xray_propagation_direction,
    wavelength,
    polarization_vector)

rotation_angle = 5.0 * np.pi / 180.
rotation_axis = np.array([0, 0, 1])
translation = np.array([0, 0, 0])
motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

print("Diffraction computations (wide detector):")
peaks_dict = polycrystal.diffract(beam, motion, detector=detector_wide)

# Extract peak positions
import torch
peaks = peaks_dict["peaks"]
columns = peaks_dict["columns"]
if torch.is_tensor(peaks):
    peaks_np = peaks.cpu().numpy()
else:
    peaks_np = np.array(peaks)

h_idx = columns.index("h")
k_idx = columns.index("k")
l_idx = columns.index("l")
tth_idx = columns.index("2theta")
# Get scattered beam direction to compute detector intersection
kout_x_idx = columns.index("K_out_x")
kout_y_idx = columns.index("K_out_y")
kout_z_idx = columns.index("K_out_z")
source_x_idx = columns.index("Source_x")
source_y_idx = columns.index("Source_y")
source_z_idx = columns.index("Source_z")

# Compute detector intersection for each peak
# Ray: P = Source + t * K_out, detector at x = detector_distance
# t = (detector_distance - Source_x) / K_out_x
det_y_list = []
det_z_list = []
for i in range(len(peaks_np)):
    kout = np.array([peaks_np[i, kout_x_idx], peaks_np[i, kout_y_idx], peaks_np[i, kout_z_idx]])
    source = np.array([peaks_np[i, source_x_idx], peaks_np[i, source_y_idx], peaks_np[i, source_z_idx]])
    if abs(kout[0]) > 1e-10:  # Avoid division by zero
        t = (detector_distance - source[0]) / kout[0]
        det_y = source[1] + t * kout[1]
        det_z = source[2] + t * kout[2]
    else:
        det_y = det_z = np.nan
    det_y_list.append(det_y)
    det_z_list.append(det_z)

det_y_arr = np.array(det_y_list)
det_z_arr = np.array(det_z_list)

print(f"\nFound {len(peaks_np)} peaks")
print("\nFirst 5 peaks (hkl, detector y, detector z, 2theta):")
for i in range(min(5, len(peaks_np))):
    hkl = peaks_np[i, h_idx:l_idx+1].astype(int)
    det_y = det_y_arr[i]
    det_z = det_z_arr[i]
    tth = np.degrees(peaks_np[i, tth_idx])
    print(f"  {hkl} -> y={det_y:.3f}, z={det_z:.3f}, 2θ={tth:.2f}°")

# ============================================================
# STEP 2: Create zoomed detector centered on a specific peak
# ============================================================
# Find a peak with valid detector position
valid_mask = ~np.isnan(det_y_arr)
if not np.any(valid_mask):
    raise ValueError("No valid peaks found!")

peak_idx = np.where(valid_mask)[0][0]  # Use first valid peak
target_y = det_y_arr[peak_idx]
target_z = det_z_arr[peak_idx]
target_hkl = peaks_np[peak_idx, h_idx:l_idx+1].astype(int)

print(f"\n=== Zooming in on peak {target_hkl} at (y={target_y:.3f}, z={target_z:.3f}) ===")

# Smaller detector (zoom factor 8x)
zoom_factor = 8
zoomed_detector_size = detector_size / zoom_factor
zoomed_pixel_size = pixel_size / 2  # Higher resolution

# Center the zoomed detector on the target peak
det_corner_0_zoom = np.array([detector_distance, target_y - zoomed_detector_size/2, target_z - zoomed_detector_size/2])
det_corner_1_zoom = np.array([detector_distance, target_y + zoomed_detector_size/2, target_z - zoomed_detector_size/2])
det_corner_2_zoom = np.array([detector_distance, target_y - zoomed_detector_size/2, target_z + zoomed_detector_size/2])

detector_zoomed = Detector(
    det_corner_0=det_corner_0_zoom,
    det_corner_1=det_corner_1_zoom,
    det_corner_2=det_corner_2_zoom,
    pixel_size=zoomed_pixel_size)

print(f"Wide detector: {detector_size:.3f} x {detector_size:.3f}, pixel={pixel_size:.6f}")
print(f"Zoomed detector: {zoomed_detector_size:.3f} x {zoomed_detector_size:.3f}, pixel={zoomed_pixel_size:.6f}")

# Re-run diffraction with zoomed detector
print("\nDiffraction computations (zoomed detector):")
peaks_dict_zoomed = polycrystal.diffract(beam, motion, detector=detector_zoomed)
print(f"Peaks in zoomed view: {len(peaks_dict_zoomed['peaks'])}")

# ============================================================
# STEP 3: Render with all three methods on both detectors
# ============================================================

print("\n--- Wide detector rendering ---")
diffraction_wide_macro = detector_wide.render(peaks_dict, frames_to_render=1, method="macro")
diffraction_wide_micro = detector_wide.render(peaks_dict, frames_to_render=1, method='micro')
diffraction_wide_nano = detector_wide.render(peaks_dict, frames_to_render=1, method='nano')

print("\n--- Zoomed detector rendering ---")
diffraction_zoom_macro = detector_zoomed.render(peaks_dict_zoomed, frames_to_render=1, method="macro")
diffraction_zoom_micro = detector_zoomed.render(peaks_dict_zoomed, frames_to_render=1, method='micro')
diffraction_zoom_nano = detector_zoomed.render(peaks_dict_zoomed, frames_to_render=1, method='nano')

# Convert to numpy for plotting
def to_numpy(arr):
    if hasattr(arr, 'cpu'):
        return arr[0].cpu().numpy()
    return np.array(arr[0])

wide_macro_np = to_numpy(diffraction_wide_macro)
wide_micro_np = to_numpy(diffraction_wide_micro)
wide_nano_np = to_numpy(diffraction_wide_nano)
zoom_macro_np = to_numpy(diffraction_zoom_macro)
zoom_micro_np = to_numpy(diffraction_zoom_micro)
zoom_nano_np = to_numpy(diffraction_zoom_nano)

# ============================================================
# STEP 4: Plot comparison
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Convert physical coordinates to pixel coordinates
# Detector renders: column increases with y, row increases with z
# Physical y=-size/2 -> column 0, y=+size/2 -> column N
# Physical z=-size/2 -> row 0, z=+size/2 -> row N
target_col = (target_y + detector_size/2) / pixel_size  # column in image
target_row = (target_z + detector_size/2) / pixel_size  # row in image (NO flip!)

# Rectangle for zoomed region (in pixel coordinates)
# Rectangle corner is at top-left of the box, centered on target
half_box = (zoomed_detector_size / pixel_size) / 2
rect_col = target_col - half_box
rect_row = target_row - half_box
rect_size = zoomed_detector_size / pixel_size

from matplotlib.patches import Rectangle

# Wide detector views (top row)
axes[0, 0].imshow(wide_macro_np, cmap='gray')
axes[0, 0].set_title("Wide: Macro (3D volume)")
axes[0, 0].set_xlabel(f"Peaks: {len(peaks_dict['peaks'])}")

# Add rectangle to mark zoomed region on all wide views
for ax in axes[0, :]:
    rect = Rectangle((rect_col, rect_row), rect_size, rect_size, 
                      linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

axes[0, 1].imshow(wide_micro_np, cmap='gray')
axes[0, 1].set_title("Wide: Micro (Gaussian)")

axes[0, 2].imshow(wide_nano_np, cmap='gray')
axes[0, 2].set_title("Wide: Nano (Airy disk)")

# Zoomed detector views (bottom row)
axes[1, 0].imshow(zoom_macro_np, cmap='gray')
axes[1, 0].set_title(f"Zoomed {zoom_factor}x: Macro")
axes[1, 0].set_xlabel(f"Peak {target_hkl}")

axes[1, 1].imshow(zoom_micro_np, cmap='gray')
axes[1, 1].set_title(f"Zoomed {zoom_factor}x: Micro")

axes[1, 2].imshow(zoom_nano_np, cmap='gray')
axes[1, 2].set_title(f"Zoomed {zoom_factor}x: Nano")

plt.suptitle(f"Single Crystal Diffraction: Wide vs Zoomed View\nTarget: {target_hkl} reflection", fontsize=14)
plt.tight_layout()
plt.show()
