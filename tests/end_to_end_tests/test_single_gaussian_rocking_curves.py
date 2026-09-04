import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from xrd_simulator.phase import Phase
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.detector import Detector
from xrd_simulator.gaussian_crystal_model import GaussianGrainish, GaussianPolycrystal
from xrd_simulator.beam import GaussianBeam
from xrd_simulator.utils import ensure_torch
from xfab.tools import form_b_mat

### Parameters
wavelength=1.0
beam_half_edgewidth = 500.0
hkl_tuple = (1, 2, 0)
detector_distance = 1e4
pixelsize =  3.0
n_pixels = 512

rocking_axis = np.array([0, 1, 0])
rocking_angle = 10 * np.pi / 180
rocking_steps = 256

polarization_vector = np.array([0, 1, 0])
eta = np.pi / 2


from xrd_simulator.phase import Phase
quartz = Phase(
   unit_cell=[4.926, 4.926, 5.4189, 90.0, 90.0, 120.0],
   sgname="P3221",  # (Quartz)
   path_to_cif_file=None,  # phases can be defined from crystalographic information files
)

beam = GaussianBeam(
    xray_propagation_direction=np.array([1.0, 0.0, 0.0]),
    beam_centroid_position=np.array([0.0, 0.0, 0.0,]),
    wavelength=wavelength,
    polarization_vector=polarization_vector,
)


# Basic crystallography
B = form_b_mat(quartz.unit_cell)
G = B @ hkl_tuple
q_norm = np.linalg.norm(G)
theta = np.arcsin(wavelength * np.linalg.norm(q_norm) / 4 / np.pi)


def make_detector(eta):

    detetctor_halfsize = n_pixels * pixelsize / 2
    detector_mid = np.array([0.0, detector_distance*np.tan(2 * theta)*np.cos(eta), detector_distance*np.tan(2 * theta)*np.sin(eta)])
    print(detector_mid)
    detector = Detector(
       det_corner_0=detector_mid + np.array([detector_distance, -1.0*detetctor_halfsize, -1.0*detetctor_halfsize]), 
       det_corner_1=detector_mid + np.array([detector_distance, 1.0*detetctor_halfsize, -1.0*detetctor_halfsize]),
       det_corner_2=detector_mid + np.array([detector_distance, -1.0*detetctor_halfsize, 1.0*detetctor_halfsize]),
       pixel_size=(pixelsize, pixelsize),
    )

    return detector


def align_grain(polycrystal, eta, ):

    B = form_b_mat(quartz.unit_cell)
    
    q = np.array(polycrystal.orientaions[0]) @ B @ hkl_tuple
    print(q)
    theta = np.arcsin(wavelength * np.linalg.norm(q) / 4 / np.pi)
    target_orientation = np.array([-np.sin(theta), np.cos(theta)*np.cos(eta), np.cos(theta)*np.sin(eta)])
    print(target_orientation)
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

    q = np.array(polycrystal.orientaions[0]) @ B @ hkl_tuple
    print(q)
    return alignment_rotation

def make_random_tensor(axis_1, axis_2):
    random_direction = np.random.normal(size=3)
    random_direction = random_direction/np.linalg.norm(random_direction)
    tensor = axis_1**2 * np.eye(3) + (axis_2**2-axis_1**2) * np.outer(random_direction, random_direction)
    return tensor


# Test independence on misorientation.
### Do simulation
motion_rock_init = RigidBodyMotion(
   rotation_axis= -rocking_axis,
   rotation_angle= 0.5 * rocking_angle,
   translation=np.array([0.0, 0.0, 0.0]),
)

motion_rock = RigidBodyMotion(
   rotation_axis=rocking_axis,
   rotation_angle= rocking_angle/rocking_steps,
   translation=np.array([0.0, 0.0, 0.0]),
)

misorientation_tensor = make_random_tensor(
    np.random.uniform(0.01, 0.01),
    np.random.uniform(0.05, 0.05),
)
shape_tensor = np.eye(3) * 70**2

polycrystal = GaussianPolycrystal(
    [GaussianGrainish(
        phase=quartz,
        position = np.zeros(3),
        shape_tensor=shape_tensor,
        orientation = R.random().as_matrix(),
        misorientation_tensor=misorientation_tensor,
        strain_tensor = np.zeros((3,3,)),
    )],
    max_misorientation = 0.1,
)

alignment_rotation = align_grain(polycrystal, eta)
polycrystal.transform(motion_rock_init, 1.0)
detector = make_detector(eta)

RSM_simulated = np.zeros((rocking_steps, n_pixels, n_pixels))

for ii in range(rocking_steps):  
    f = polycrystal.render_detector_frame(
        beam=beam,
        detector=detector,
    )
    polycrystal.transform(motion_rock, 1.0)

    

    RSM_simulated[ii] = f
   
polycrystal.transform(motion_rock_init, 1.0)


if __name__ == "__main__":

    fig, axs = plt.subplots(2,2, figsize = (8,8))
    y_lab = (np.arange(n_pixels) - n_pixels/2) * pixelsize
    z_lab = np.tan(2 * theta) * detector_distance
    eta_pixels = np.atan2(y_lab, z_lab)

    axs[0,0].imshow(np.sum(RSM_simulated, axis=1), extent = (eta_pixels[0], eta_pixels[-1], -rocking_angle / 2, rocking_angle / 2))
    axs[0,0].set_title('RSM averaged over detector z')
    axs[0,0].set_xlabel('Eta angle (apparent q_roll)')
    axs[0,0].set_ylabel('Rocking angle (q_rock)')

    from xrd_simulator.laue import _project_misorientation_tensor
    g = polycrystal.orientaions[0]
    p = torch.einsum('ij,jk,k->i', torch.Tensor(g), torch.Tensor(B), torch.Tensor(hkl_tuple))
    p = p / torch.linalg.norm(p)
    T_proj = np.array(_project_misorientation_tensor(polycrystal.misori_concentration_tensors, ensure_torch(p)[None, :]))[0]

    q_rock = np.array([np.cos(theta), 0, np.sin(theta)])
    q_roll = np.array([0, 1, 0])
    trans_mat = np.stack([q_roll, q_rock])
    print(trans_mat.shape)
    T_proj_reduced = np.einsum('ui,ij,vj->uv', trans_mat, T_proj, trans_mat)
    misori_projected_eta = 1/np.linalg.inv(T_proj_reduced)[0,0]


    prop_factor = y_lab[-1] / eta_pixels[-1]
    a = misori_projected_eta
    b = prop_factor**2 * 1/shape_tensor[0, 0]
    c = 1/ (1/a + 1/b)

    axs[0,1].plot(eta_pixels, np.sum(RSM_simulated, axis =(0,1))/np.max(np.sum(RSM_simulated, axis =(0,1))), linewidth = 4)
    axs[0,1].plot(eta_pixels, np.exp(-eta_pixels**2 * misori_projected_eta), '--')
    axs[0,1].plot(eta_pixels, np.exp(-y_lab**2 * 1/shape_tensor[0, 0]), '--')
    axs[0,1].plot(eta_pixels, np.exp(-eta_pixels**2 * c), 'k--')
    axs[0,1].legend(['Simulated angular width', 'misorientation contribution', 'grain size contribution', 'Combined'])
    axs[0,1].set_xlabel('Eta (rad)')

    z_lab = np.linspace(detector.det_corner_0[2], detector.det_corner_2[2], n_pixels)
    z_mid = 0.5 * (detector.det_corner_0[2] + detector.det_corner_2[2])
    axs[1,0].plot(z_lab, np.sum(RSM_simulated, axis =(0,2))/np.max(np.sum(RSM_simulated, axis =(0,2))), linewidth = 4)
    axs[1,0].plot(z_lab, np.exp(-(z_lab -z_mid)**2 * 1/shape_tensor[0, 0] * np.cos(2 * theta)**2), '--',  color='C2')
    axs[1,0].set_xlabel('Lab z (\u03bcm)')


    rocking_angles = np.linspace(-0.5*rocking_angle, 0.5*rocking_angle, rocking_steps, endpoint=False)
    misori_projected_rock = 1/np.linalg.inv(T_proj_reduced)[1,1]

    axs[1,1].plot(rocking_angles, np.sum(RSM_simulated, axis =(1,2))/np.max(np.sum(RSM_simulated, axis =(1,2))), linewidth = 4)
    axs[1,1].plot(rocking_angles, np.exp(-rocking_angles**2 * misori_projected_rock), '--',  color='C1')
    axs[1,1].set_xlabel('Rocking angle (q_rock)')
    plt.show()
