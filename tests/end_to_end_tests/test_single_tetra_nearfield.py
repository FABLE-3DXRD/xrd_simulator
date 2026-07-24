import numpy as np
from scipy.spatial.transform import Rotation as R

from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.beam import Beam
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.detector import Detector
from xfab.tools import form_b_mat
from xrd_simulator.gaussian_crystal_model import GaussianGrainish, GaussianPolycrystal
from xrd_simulator.beam import GaussianBeam
from xrd_simulator.utils import ensure_torch

### Parameters
wavelength=0.28523
beam_half_edgewidth = 100.0
tetrahedron_bbox_size = 150.0
eta = np.pi / 2
hkl_tuple = (2, 1, 0)
detector_distance = 0.5e4
pixelsize =  10.0
n_pixels = 2000
rocking_axis = np.array([0, 1, 0])
rocking_angle = 5.0 * np.pi / 180
strain = -0.001*np.eye(3)
strain = -0.000*np.array([[0,1,0],
                          [1,0,0],
                          [0,0,0],])
### Utility funcitons
def align_grain(polycrystal, grainindex, beam,  hkl_tuple, eta, ):

   phase_index = int(polycrystal.element_phase_map[grainindex])
   phase = polycrystal.phases[phase_index]
   B = form_b_mat(phase.unit_cell)
   
   q = np.array(polycrystal.orientation_lab[0]) @ B @ hkl_tuple
   theta = np.arcsin(beam.wavelength * np.linalg.norm(q) / 4 / np.pi)
   target_orientation = np.array([-np.sin(theta), np.cos(theta)*np.cos(eta), np.cos(theta)*np.sin(eta)])
   current_orientation = q / np.linalg.norm(q)

   rotation_to_x = R.from_euler('zx',
                              (np.arctan2(np.sqrt(current_orientation[1]**2 +current_orientation[2]**2, ), current_orientation[0]),
                                 np.arctan2(current_orientation[2], current_orientation[1]),)).inv()
   rotation_to_target = R.from_euler('zx',
                              (np.arctan2(np.sqrt(target_orientation[1]**2 +target_orientation[2]**2, ), target_orientation[0]),
                                 np.arctan2(target_orientation[2], target_orientation[1]),))

   total_rotation = (rotation_to_target*rotation_to_x)
   rot_angle = total_rotation.magnitude()
   rot_axis = total_rotation.as_rotvec() / rot_angle

   alignment_rotation = RigidBodyMotion(
      rotation_axis=rot_axis,
      rotation_angle=rot_angle,
      translation=np.array([0.0, 0.0, 0.0]),
   )

   polycrystal.transform(alignment_rotation, 1.0)
   return alignment_rotation


### Define a beam

beam_vertices = np.array(
   [
      [-1e6, -beam_half_edgewidth, -beam_half_edgewidth],
      [-1e6, beam_half_edgewidth, -beam_half_edgewidth],
      [-1e6, beam_half_edgewidth, beam_half_edgewidth],
      [-1e6, -beam_half_edgewidth, beam_half_edgewidth],
      [1e6, -beam_half_edgewidth, -beam_half_edgewidth],
      [1e6, beam_half_edgewidth, -beam_half_edgewidth],
      [1e6, beam_half_edgewidth, beam_half_edgewidth],
      [1e6, -beam_half_edgewidth, beam_half_edgewidth],
   ]
)

beam = Beam(
   beam_vertices,
   xray_propagation_direction=np.array([1.0, 0.0, 0.0]),
   wavelength=wavelength,
   polarization_vector=np.array([0.0, 1.0, 0.0]),
)

### Define sample

tetr_vertexes = np.array(
    [[tetrahedron_bbox_size, tetrahedron_bbox_size, tetrahedron_bbox_size,],
     [tetrahedron_bbox_size, -tetrahedron_bbox_size, -tetrahedron_bbox_size,],
     [-tetrahedron_bbox_size, tetrahedron_bbox_size, -tetrahedron_bbox_size,],
     [-tetrahedron_bbox_size, -tetrahedron_bbox_size, tetrahedron_bbox_size,],
     ]
)

mesh = TetraMesh.generate_mesh_from_vertices(
    tetr_vertexes, np.arange(4)[None, :],
)

quartz = Phase(
   unit_cell=[4.926, 4.926, 5.4189, 90.0, 90.0, 120.0],
   sgname="P3221",  # (Quartz)
   path_to_cif_file=None,  # phases can be defined from crystalographic information files
)

cs_cl = Phase(
   unit_cell=[4.994, 4.994, 4.994, 90.0, 90.0, 90.0],
   sgname="F432",  # (Quartz)
   path_to_cif_file=None,  # phases can be defined from crystalographic information files
)

orientation = R.random(mesh.number_of_elements).as_matrix()
element_phase_map = np.zeros(mesh.number_of_elements, dtype=int)
polycrystal = Polycrystal(
   mesh,
   orientation,
   strain=strain,
   phases=quartz,
   element_phase_map=element_phase_map,
)


### Align for single-crystal experiment
alignment_rotation = align_grain(polycrystal, 0, beam, hkl_tuple, eta)

phase_index = int(polycrystal.element_phase_map[0])
phase = polycrystal.phases[phase_index]
B = form_b_mat(phase.unit_cell) 
q = np.array(polycrystal.orientation_lab[0]) @ B @ hkl_tuple
theta = np.arcsin(beam.wavelength * np.linalg.norm(q) / 4 / np.pi)

### Place detector in the beam
detetctor_halfsize = n_pixels * pixelsize / 2
detector_mid = np.array([1.0, 0.0, 0.0]) * detector_distance

# The detector plane is defined by it's corner coordinates det_corner_0,det_corner_1,det_corner_2
detector = Detector(
   det_corner_0=detector_mid + np.array([0.0, -1.0, -1.0])*detetctor_halfsize, 
   det_corner_1=detector_mid + np.array([0.0, 1.0, -1.0])*detetctor_halfsize,
   det_corner_2=detector_mid + np.array([0.0, -1.0, 1.0])*detetctor_halfsize,
   pixel_size=(pixelsize, pixelsize),
   gaussian_sigma=1.0,
   max_gaussian_kernel_radius=5,
)

### Do simulation
motion_rock_init = RigidBodyMotion(
   rotation_axis= -rocking_axis,
   rotation_angle= 0.5 * rocking_angle,
   translation=np.array([0.0, 0.0, 0.0]),
)

motion_rock = RigidBodyMotion(
   rotation_axis=rocking_axis,
   rotation_angle= rocking_angle,
   translation=np.array([0.0, 0.0, 0.0]),
)

motion_rock_reset = RigidBodyMotion(
   rotation_axis=rocking_axis,
   rotation_angle= 0.5 * rocking_angle,
   translation=np.array([0.0, 0.0, 0.0]),
)

polycrystal.transform(motion_rock_init, 1.0)


peaks_dict = polycrystal.diffract(beam, motion_rock, detector=detector)
diffraction_pattern, peaks_dict = detector.render(
   peaks_dict, frames_to_render=0, method="macro"
)

polycrystal.transform(motion_rock_reset, 1.0)

pattern = (
diffraction_pattern[0].cpu().numpy()
if hasattr(diffraction_pattern, "cpu")
else diffraction_pattern[0]
)

############# Gaussian based workflow ###############
def make_random_tensor(axis_1, axis_2):
    random_direction = np.random.normal(size=3)
    random_direction = random_direction/np.linalg.norm(random_direction)
    tensor = axis_1**2 * np.eye(3) + (axis_2**2-axis_1**2) * np.outer(random_direction, random_direction)
    return tensor


misorientation_tensor = make_random_tensor(
    np.random.uniform(0.005, 0.005),
    np.random.uniform(0.0001, 0.0001),
)

grain_list = []
for ii, vert1 in enumerate(tetr_vertexes):
   for vert2 in tetr_vertexes[ii+1:]:

      grain = GaussianGrainish(
                     phase=quartz, #  For now it assumes all gaussians are the same phase, but it just needs a wrapper for multiphase
                     position=0.3*(vert1+vert2), # 3 vector centroid real-space position
                     shape_tensor=tetrahedron_bbox_size**2*np.eye(3)/8 + np.outer(vert1-vert2, vert1-vert2)/8, # 3-by-3 symmetric shape tensor where the eigenvalues are the radii-squared.
                     orientation=orientation[0], # 3-by-3 rotation matrix.
                     misorientation_tensor=misorientation_tensor, # 3-by-3 misorientation tensor where the eigenvalues are the misorientaion spread in radians squared.
                                                                     # misorientation vectors live in laboratory coordinates.
                     strain_tensor=strain # 3-by-3 symmetric strain tensor.
                  )
      grain_list.append(grain)

grain = GaussianGrainish(
        phase=quartz, #  For now it assumes all gaussians are the same phase, but it just needs a wrapper for multiphase
        position=np.zeros(3), # 3 vector centroid real-space position
        shape_tensor=tetrahedron_bbox_size**2*np.eye(3)/4, # 3-by-3 symmetric shape tensor where the eigenvalues are the radii-squared.
        orientation=orientation[0], # 3-by-3 rotation matrix.
        misorientation_tensor=misorientation_tensor, # 3-by-3 misorientation tensor where the eigenvalues are the misorientaion spread in radians squared.
                                                        # misorientation vectors live in laboratory coordinates.
        strain_tensor=strain # 3-by-3 symmetric strain tensor.
    )
grain_list.append(grain)

gauss_polycrystal = GaussianPolycrystal(grain_list, max_misorientation = 0.005)

gaussian_beam = GaussianBeam(
    xray_propagation_direction=np.array([1.0, 0.0, 0.0]),
    beam_centroid_position=np.array([0.0, 0.0, 0.0,]),
    wavelength=beam.wavelength,
    long_axis_width = beam_half_edgewidth,
)

gauss_polycrystal.transform(alignment_rotation)


f = gauss_polycrystal.render_detector_frame(
    beam=gaussian_beam,
    detector=detector,
    sample_rotation_during_exposure = rocking_axis * rocking_angle /2.5,
    timing=True,
)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # render returns (frames, height, width), take first frame

   #  img = axs[0].imshow(np.log10(pattern+1e0), cmap="jet")
    img = axs[0].imshow(pattern, vmin =0, vmax = 5e6, cmap="jet")
    axs[0].set_title('Tetrahedron based model')
    axs[0].grid()

    img = axs[1].imshow(np.log10(f+1e0), cmap="jet")
    img = axs[1].imshow(f, vmin =0, vmax = 2e4, cmap="jet")
    axs[1].set_title('Gaussian based_model')
    axs[1].grid()
    plt.show()
