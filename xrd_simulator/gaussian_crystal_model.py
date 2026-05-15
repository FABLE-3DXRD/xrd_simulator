from scipy.spatial.transform import Rotation as R
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from xrd_simulator.phase import Phase
from xrd_simulator.utils import ensure_torch
from xrd_simulator.detector import Detector

from xfab.tools import form_a_mat, genhkl_base
from xfab import sg


class GaussianGrainish:
    """It's not a grain, it's a grain-ish.
    """

    def __init__(self,
        phase: Phase,
        position: npt.NDArray | Tensor = np.array([0, 0, 0]), 
        shape_tensor: npt.NDArray | Tensor = np.eye(3),
        orientation: npt.NDArray | Tensor = np.eye(3), # 3by3 Rotation matrix.
        misorientation_tensor: npt.NDArray | Tensor = 0.0175**2 * np.eye(3), # Default one degree isotropic
        strain_tensor: npt.NDArray | Tensor = np.zeros((3, 3,)),
    ):
        
        self.phase = phase
        self.position = position
        self.shape_tensor = shape_tensor
        self.orientation = orientation
        self.misorientation_tensor = misorientation_tensor
        self.strain_tensor = strain_tensor


levi_cita_symbol = np.zeros((3,3,3))
levi_cita_symbol[0, 1, 2] = 1
levi_cita_symbol[1, 2, 0] = 1
levi_cita_symbol[2, 0, 1] = 1
levi_cita_symbol[0, 2, 1] = -1
levi_cita_symbol[1, 0, 2] = -1
levi_cita_symbol[2, 1, 0] = -1

levi_cita_symbol = torch.tensor(levi_cita_symbol)


class GaussianPolycrystal:

    def __init__(self,
        grain_list: list[GaussianGrainish],
    ):
        
        phases_list = list(set([grain.phase for grain in grain_list]))
        n_phases = len(phases_list)
        assert n_phases == 1

        self.n_grains = len(grain_list)
        self.phase = phases_list[0]

        self.positions = torch.stack([ensure_torch(grain.position) for grain in grain_list])
        self.shape_concentration_tensors = torch.stack([ensure_torch(np.linalg.inv(grain.shape_tensor)) for grain in grain_list])

        self.orientaions = torch.stack([ensure_torch(grain.orientation) for grain in grain_list])
        self.misori_concentration_tensors = torch.stack([ensure_torch(np.linalg.inv(grain.misorientation_tensor)) for grain in grain_list])
        
        self.strains = torch.stack([ensure_torch(grain.strain_tensor) for grain in grain_list])

    def render_detector_frame(
        self,
        detector: Detector,
        xray_propagation_direction: npt.NDArray | Tensor,
        wavelength: float,
        sample_orientation: npt.NDArray | Tensor = np.eye(3),
        sample_rotation_during_exposure: npt.NDArray | Tensor = np.zeros(3),
        max_misorientation: float = 0.1,
    ):

        sample_orientation = ensure_torch(sample_orientation)
        sample_rotation_during_exposure = ensure_torch(sample_rotation_during_exposure)

        #Rotate detector and incident beam by inverse of sample-rotation.
        xray_propagation_direction = torch.einsum('ij,i->j', sample_orientation, ensure_torch(xray_propagation_direction) )
        detector_norm = torch.einsum('ij,i->j', sample_orientation, ensure_torch(np.cross(detector.W[0,:], detector.W[1,:])))
        detector_origin = torch.einsum('ij,i->j', sample_orientation, ensure_torch(detector.ori))
        W = torch.einsum('ij,ui->uj', sample_orientation, ensure_torch(detector.W))
        pixellengths = torch.tensor([detector.pixel_size_y, detector.pixel_size_z])
        
        # Simulate sample-rotation by adding a rotation to the grain misorientation
        rotation_vector = torch.einsum('ij,i->j', sample_orientation, sample_rotation_during_exposure)
        smeared_misorientation_tensors = torch.linalg.inv(torch.linalg.inv(self.misori_concentration_tensors) + torch.outer(rotation_vector, rotation_vector))

        #Figure out what reflections are in the bragg-condition
        A = form_a_mat(self.phase.unit_cell)
        B = 2 * np.pi * np.linalg.inv(A).T # TODO Check if the 2 pi is conventional here 

        # Generate structure-factors and 
        max_angle = detector._get_wrapping_cone(xray_propagation_direction, np.mean([0, 0, 0]))
        self.phase._setup_diffracting_planes(wavelength=wavelength, min_bragg_angle=0.0, max_bragg_angle=max_angle)  #TODO Using private method
        sg_obj = sg.sg(sgname=self.phase.sgname)

        hkl_list = genhkl_base(
            self.phase.unit_cell,
            sg_obj.syscond,
            0.0, np.sin(max_angle) / wavelength,
            sg_obj.crystal_system,
            sg_obj.Laue,
        )
        self.phase._set_structure_factors(hkl_list)  #TODO Using private method
        structurefactors = ensure_torch(np.sum(self.phase.structure_factors**2, axis=1))

        uv_corrds_list = []
        splat_concentration_tensors_list = []
        scalefactors_list = []

        # Loop over reflection orders
        for hkl, S in zip(hkl_list, structurefactors):

            # TODO reduce the number of symmetries evaluated for low-multiplicity peaks
            n_symmetries = len(self.phase.rot)


            # Quick test to discard reflections far from Bragg-condition
            h = torch.tensor(B @ hkl)
            h_norm = torch.linalg.norm(h)
            theta_angle_unstrained = np.asin( h_norm * wavelength / 4 / np.pi )
            p_vectors = torch.einsum('ghi,gij,sjk,k->gsh', torch.eye(3)[None,:,:] - self.strains, self.orientaions, ensure_torch(self.phase.rot), h)
            p_vectors_norm = torch.linalg.norm(p_vectors, axis=-1)
            dp = torch.einsum('i,gsi->gs', xray_propagation_direction, p_vectors) / p_vectors_norm            
            does_diffract = torch.abs( dp + torch.sin(theta_angle_unstrained) ) <  torch.abs(3 * max_misorientation * torch.cos(theta_angle_unstrained))

            # Select the relevant reflections and flatten grain- and symetry-indexes.
            orientations = torch.tile(self.orientaions[:, None, :, :], (1, n_symmetries, 1, 1))[does_diffract]
            misori_concentration_tensors = torch.tile(smeared_misorientation_tensors[:, None, :, :], (1, n_symmetries, 1, 1))[does_diffract]
            p_vectors = p_vectors[does_diffract]

            # Do pole-figure part of the calculation
            mean_scattering_directions, partiality, azim_direction, azim_width = self.get_diffraction_arcsegment(
                orientations, p_vectors, misori_concentration_tensors, xray_propagation_direction, wavelength
            )

            # Splat grain realspace shapes (Consider using the non-strained non-azimuthally shifted directions to simplify gradients later)
            shape_concentration_tensors = torch.tile(self.shape_concentration_tensors[:, None, :, :], (1, n_symmetries, 1, 1))[does_diffract]
            detectorspace_grainshape_projections, projected_thicknes_scale_factor = self.splat_grainshapes(
                mean_scattering_directions,
                shape_concentration_tensors,
                W,
                pixellengths,
            )

            # Direct beam is pos + mean_scatt_dir * x, detector is det_ori dot det_norm = 0 
            pos = torch.tile(self.positions[:, None, :], (1, n_symmetries, 1,))[does_diffract]
            ray_lengths = torch.einsum('xi,i->x', detector_origin[None, :] - pos, detector_norm) / torch.einsum('xi,i->x', mean_scattering_directions, detector_norm)
            point_of_detector_intersection = pos + ray_lengths[:,None] * mean_scattering_directions
            uv_coords = torch.einsum('xi, vi, v->xv',point_of_detector_intersection - detector_origin[None, :], W, 1/pixellengths)

            #Do smearing due to angular divergence.
            azimuthal_direction_uv = torch.einsum('xi,ui->xu', azim_direction, W) / pixellengths[None, :] * ray_lengths[:, None] * azim_width[:, None] / (1 - torch.einsum('xi,ui->xu', mean_scattering_directions, W)**2)
            azimuthal_smearing_tensor = torch.einsum('xu,xv->xuv',azimuthal_direction_uv, azimuthal_direction_uv)
            detspace_splat_concentration = torch.linalg.inv( torch.linalg.inv(detectorspace_grainshape_projections) + azimuthal_smearing_tensor)
            intensity_spread_out_factor = torch.sqrt( torch.linalg.det(detspace_splat_concentration) / torch.linalg.det(detectorspace_grainshape_projections) )  #TODO This was (organically) vibe-coded. Check on paper if there is a simplification.
            

            uv_corrds_list.append(uv_coords)
            scalefactors_list.append(S * projected_thicknes_scale_factor * partiality * intensity_spread_out_factor)
            splat_concentration_tensors_list.append(detspace_splat_concentration)
                         

        f = self.rasterize_on_detector(
            torch.concat(uv_corrds_list),
            torch.concat(scalefactors_list),
            torch.concat(splat_concentration_tensors_list),
            detector.shape,
        )

        return f
    
    def splat_grainshapes(
            self,
            mean_scattering_directions: Tensor,
            shape_concentration_tensors: Tensor,
            W: Tensor,
            pixellengths: Tensor,
    ):
        
        # grain_volume = torch.sqrt(1/torch.linalg.det(shape_concentration_tensors))
        dSd = torch.einsum('xi,xij,xj->x', mean_scattering_directions, shape_concentration_tensors, mean_scattering_directions)

        inner_term = shape_concentration_tensors - torch.einsum(
            'xij,xj,xk,xkl->xil',
            shape_concentration_tensors,
            mean_scattering_directions,
            mean_scattering_directions,
            shape_concentration_tensors,
        ) / dSd[:, None, None]
        
        W_scaled = W * pixellengths[:, None]
        projected_shape_pixelunits = torch.einsum(
            'ui,xij,vj->xuv', W_scaled, inner_term, W_scaled, 
        )

        return projected_shape_pixelunits, 1/torch.sqrt(dSd)
            
    def rasterize_on_detector(
            self,
            uv_corrds: Tensor,
            scale_factors: Tensor,
            concentration_tensors: Tensor,
            shape: tuple,
            patch_size: int = 16,
            splat_max_size: float = 300.0
        ):


        u, v = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        f = torch.zeros(shape)

        patch_size = 16
        n_patches_dim1 = (shape[0]-1)//patch_size+1
        n_patches_dim2 = (shape[1]-1)//patch_size+1

        for patch_index_1 in range(n_patches_dim1):
            for patch_index_2 in range(n_patches_dim2):

                patch_slice = (slice(patch_size*patch_index_1, patch_size*(patch_index_1+1)),
                               slice(patch_size*patch_index_2, patch_size*(patch_index_2+1)),)

                patch_mean_u = patch_size*(patch_index_1+0.5)
                patch_mean_v = patch_size*(patch_index_2+0.5)
                
                include_index = (uv_corrds[:, 0]-patch_mean_u)**2 + (uv_corrds[:, 1]-patch_mean_v)**2 < splat_max_size**2
                if not torch.any(include_index):
                    continue


                local_coords = torch.stack([u[patch_slice][None, :, :] - uv_corrds[include_index, 0, None, None],
                                            v[patch_slice][None, :, :] - uv_corrds[include_index, 1, None, None],
                                            ], axis=1)

                f[patch_slice] +=  torch.sum(scale_factors[include_index, None, None]\
                    * torch.exp(- torch.einsum('xiuv,xij,xjuv->xuv' ,local_coords, concentration_tensors[include_index, :, :], local_coords)), axis=0)

        return f

    def get_diffraction_arcsegment(self, oris, p_vectors, misori_tensors, xray_propagation_direction, wavelength):

        # Splat onto poelfigure
        p_norm = torch.linalg.norm(p_vectors, axis=-1)
        D = torch.einsum('xi,xij,xj->x', p_vectors, misori_tensors, p_vectors)/ p_norm**2

        inner_part = misori_tensors - torch.einsum(
            'xij,xj,xk,xkl->xil',
            misori_tensors,
            p_vectors,
            p_vectors,
            misori_tensors,
        ) / D[:, None, None] / p_norm[:, None, None]**2
        T_proj = torch.einsum(
            'xj,ijk,xil,lmn,xm->xkn',
            p_vectors,
            levi_cita_symbol,
            inner_part,
            levi_cita_symbol,
            p_vectors,
        ) / p_norm[:, None, None]**2

        # Compute point of "exact bragg condition" in the plane Span(k_0, p)
        theta_angle = np.asin( p_norm * wavelength / 4 / np.pi )
        dir_scatteringplane_norm = (p_vectors - xray_propagation_direction[None, :] * np.einsum('xi,i->x', p_vectors, xray_propagation_direction)[:, None] )
        dir_scatteringplane_norm = dir_scatteringplane_norm / torch.linalg.norm(dir_scatteringplane_norm, axis=-1)[:, None]
        q_0_unit = torch.cos(theta_angle)[:, None] * dir_scatteringplane_norm - torch.sin(theta_angle)[:, None] * xray_propagation_direction[None, :]
        dir_scatteringplane_orth = torch.einsum('ijk,j,xk->xi', levi_cita_symbol, xray_propagation_direction, dir_scatteringplane_norm)

        # Compute partiality, mean direction, and azimthal spread
        A = torch.einsum('xi,xij,xj->x', q_0_unit, T_proj, q_0_unit,)
        B = torch.einsum('xi,xij,xj->x', q_0_unit, T_proj, dir_scatteringplane_orth,)
        C = torch.einsum('xi,xij,xj->x', dir_scatteringplane_orth, T_proj, dir_scatteringplane_orth,)
        
        azim_divergence = np.sqrt(1 / C) * torch.sin(theta_angle) # TODO Check trigonometric function here
        azim_offset = B / C
        mean_scattering_directions = torch.cos(2*theta_angle)[:, None] * xray_propagation_direction[None, :]\
            + torch.sin(2*theta_angle)[:, None]*(torch.cos(azim_offset)[:, None]*dir_scatteringplane_norm + torch.sin(azim_offset)[:, None]*dir_scatteringplane_orth)
        partiality = torch.exp(-A + B**2 / C) * 2 * torch.sqrt( torch.linalg.det(misori_tensors) / D ) / torch.sqrt(C)

        return mean_scattering_directions, partiality, dir_scatteringplane_orth, azim_divergence 

    def render_polefigure(
        self,
        hkl: tuple[int],
        resolution_in_degrees: float = 1.0,
        both_hemispheres: bool = False,
        max_misorientation: float = 0.1,
    ):
        
        # Make coordinate arrays
        if both_hemispheres:
            polar, azim = np.meshgrid(np.linspace(0, np.pi, int(180//resolution_in_degrees)+1),
                                      np.linspace(0, 2*np.pi, int(360//resolution_in_degrees)+1))
        else:
            polar, azim = np.meshgrid(np.linspace(0, np.pi/2, int(90//resolution_in_degrees)+1),
                                      np.linspace(0, 2*np.pi, int(360//resolution_in_degrees)+1))
            
        y_map = torch.tensor(np.stack([
            np.sin(polar) * np.cos(azim),
            np.sin(polar) * np.sin(azim),
            np.cos(polar)
            ], axis=-1))
        
        p, scale, T_proj = self.splat_onto_polefigure(hkl)
        patch_size = 16
        
        f = self.rasterize_on_unitvector_map(
            y_map,
            p,
            scale,
            T_proj,
            max_angle= 3*max_misorientation + (resolution_in_degrees*np.pi/180)*patch_size/2,
        )

        return f, polar, azim

    def splat_onto_polefigure(
            self,
            hkl: tuple[int],
        ):


        A = form_a_mat(self.phase.unit_cell)
        B = 2 * np.pi * np.linalg.inv(A).T
        h = torch.tensor(B @ hkl)
        h = h / torch.linalg.norm(h)

        # TODO reduce the number of symmetries evaluated for low-multiplicity peaks

        n_symmetries = len(self.phase.rot)
        
        volumes = torch.sqrt(torch.linalg.det(self.shape_concentration_tensors))
        p_vectors = torch.einsum('gij,sjk,k->gsi', self.orientaions, ensure_torch(self.phase.rot), h)        
        
        # This is the trick:
        pTp = torch.einsum('gsi,gij,gsj->gs', p_vectors, self.misori_concentration_tensors, p_vectors)
        inner_part = self.misori_concentration_tensors[:, None, :, :] - torch.einsum(
            'gij,gsj,gsk,gkl->gsil',
            self.misori_concentration_tensors,
            p_vectors,
            p_vectors,
            self.misori_concentration_tensors,
        ) / pTp[:, :, None, None]
        projected_misorientation = torch.einsum(
            'gsj,ijk,gsil,lmn,gsm->gskn',
            p_vectors,
            levi_cita_symbol,
            inner_part,
            levi_cita_symbol,
            p_vectors,
        )
        
        scale = 1 / n_symmetries / torch.sum(volumes) * volumes[:, None] * 2 * torch.sqrt(torch.linalg.det(self.misori_concentration_tensors))[:, None] / np.sqrt( pTp )
        
        return p_vectors, scale, projected_misorientation                

    def rasterize_on_unitvector_map(
        self,
        y : Tensor,
        p : Tensor,
        scale : Tensor,
        T_proj : Tensor,
        max_angle: float,
        patch_size: int = 16,
    ):
        
        shape = y.shape[:2]
        min_dp = np.cos(max_angle)
        n_patches_dim1 = (shape[0]-1)//patch_size+1
        n_patches_dim2 = (shape[1]-1)//patch_size+1

        # Rasterization
        f = torch.zeros(shape)

        for patch_index_1 in range(n_patches_dim1):
            for patch_index_2 in range(n_patches_dim2):

                # Figure out what splat lie in this pole figure patch
                y_patch = y[patch_size*patch_index_1:patch_size*(patch_index_1+1),
                            patch_size*patch_index_2:patch_size*(patch_index_2+1)]
                patch_mean = torch.mean(y_patch, axis=(0, 1))
                patch_mean_y = patch_mean / torch.linalg.norm(patch_mean)
                include_index = torch.abs(torch.einsum('gsj,j->gs', p, patch_mean_y)) > min_dp
                                # If none, continue
                if not torch.any(include_index):
                    continue

                # Evaluate gaussians
                arg = -torch.einsum('pai,xij,paj->xpa', y_patch, T_proj[include_index], y_patch)
                vals = torch.exp(arg) * scale[include_index, np.newaxis, np.newaxis]
                
                f[patch_size*patch_index_1:patch_size*(patch_index_1+1),
                  patch_size*patch_index_2:patch_size*(patch_index_2+1)]\
                    += torch.sum(vals, axis=0)
        
        return f
