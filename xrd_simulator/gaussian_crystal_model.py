from scipy.spatial.transform import Rotation as R
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from xrd_simulator.phase import Phase
from xrd_simulator.utils import ensure_torch
from xrd_simulator.detector import Detector
from xrd_simulator.laue import _get_diffraction_arcsegment
from xrd_simulator.beam import GaussianBeam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.scattering_factors import _polarization

from xfab.tools import form_b_mat, genhkl_base
from xfab import sg


import time

class GaussianGrainish:
    """It's not a grain, it's a grain-ish.
    
    A grain-ish has a 3D-gaussian density distribution and a narrow (<=0.1 rad) 3D
    orientation distribution function.

    Attributes
    ----------
    phase : Phase
        Object representing the crystal structure of the grain.
    position : torch.Tensor | np.array
        Position of the grain centroid, shape ``(3,)``
    shape_tensor : torch.Tensor | np.array
        Covariance tensor of the real-space grain shape. ``(3, 3,)``.
    orientation : torch.Tensor | np.array
        Orientation of the centroid of the ODF as a rotation matrix, shape ``(3, 3,)``
    misorientation_tensor : torch.Tensor | np.array
        Covariance tensor of the grain orientation in the left-hand tangent space. 
        aka. laboratory coordinates, shape ``(3, 3,)``
    strain_tensor : torch.Tensor | np.array
        Strain tensor in laboratory coordinates, shape ``(3, 3,)``
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

class GaussianPolycrystal:

    def __init__(self,
        grain_list: list[GaussianGrainish],
        max_grain_size: float = 1000.0,
        max_misorientation: float = 0.1, 
    ):
        
        phases_list = list(set([grain.phase for grain in grain_list]))
        n_phases = len(phases_list)
        assert n_phases == 1

        self.n_grains = len(grain_list)
        self.max_grain_size = max_grain_size
        self.max_misorientation = max_misorientation
        self.phase = phases_list[0]

        self.positions = torch.stack([ensure_torch(grain.position) for grain in grain_list])
        self.shape_concentration_tensors = torch.stack([ensure_torch(np.linalg.inv(grain.shape_tensor)) for grain in grain_list])

        self.orientaions = torch.stack([ensure_torch(grain.orientation) for grain in grain_list])
        self.misori_concentration_tensors = torch.stack([ensure_torch(np.linalg.inv(grain.misorientation_tensor)) for grain in grain_list])
        
        self.strains = torch.stack([ensure_torch(grain.strain_tensor) for grain in grain_list])

    def render_detector_frame(
        self,
        beam: GaussianBeam,
        detector: Detector,
        sample_orientation: npt.NDArray | Tensor = np.eye(3),
        sample_translation: npt.NDArray | Tensor = np.zeros(3),
        sample_rotation_during_exposure: npt.NDArray | Tensor = np.zeros(3),
        timing=False
    ):

        if timing:
            t0 = time.time()

        xray_propagation_direction = beam.xray_dir
        wavelength = beam.wavelength
        sample_orientation = ensure_torch(sample_orientation)
        sample_rotation_during_exposure = ensure_torch(sample_rotation_during_exposure)
        
        # Rotate detector and incident beam by inverse of sample-rotation.
        xray_propagation_direction = torch.einsum('ij,i->j', sample_orientation, ensure_torch(xray_propagation_direction) )
        detector_origin = detector.pixel_coordinates[0,0]
        W = np.stack([detector.zdhat, detector.ydhat])
        detector_norm = torch.einsum('ij,i->j', sample_orientation, ensure_torch(np.cross(W[0,:], W[1,:])))
        detector_origin = torch.einsum('ij,i->j', sample_orientation, ensure_torch(detector_origin))
        W = torch.einsum('ij,ui->uj', sample_orientation, ensure_torch(W))
        pixellengths = torch.tensor([detector.pixel_size_y, detector.pixel_size_z])

        #Compute intersection of grains and beam
        intersection_pos, intersection_shape_concentration_tensors, beam_intensity_factors, grains_hit\
            = beam._intersect(
            ensure_torch(self.positions),
            ensure_torch(self.shape_concentration_tensors),
            sample_orientation,
            sample_translation,
            self.max_grain_size,
        )

        if timing:
            print(f'Beam-grain intersection took {time.time()-t0}.')
            t0 = time.time()

        # Simulate sample-rotation by adding a rotation to the grain misorientation
        sample_rotation_during_exposure = ensure_torch(sample_rotation_during_exposure)
        rotation_vector = torch.einsum('ij,i->j', sample_orientation, sample_rotation_during_exposure)
        smeared_misorientation_tensors = torch.linalg.inv(torch.linalg.inv(self.misori_concentration_tensors[grains_hit]) + torch.outer(rotation_vector, rotation_vector))

        #Construct some crystal and geometry information.
        B = torch.Tensor(form_b_mat(self.phase.unit_cell))
        max_angle = detector._get_wrapping_cone(xray_propagation_direction, np.mean([0, 0, 0]))
        self.phase._setup_diffracting_planes(wavelength=wavelength, min_bragg_angle=0.0, max_bragg_angle=max_angle+0.1)  #TODO Using private method
        
        # Get miller indicies and structure factors
        miller_indices = torch.Tensor(self.phase.miller_indices)
        if self.phase.structure_factors is not None:
            structure_factors = torch.sum(
                ensure_torch(self.phase.structure_factors) ** 2, axis=1
            )
            miller_indices = miller_indices[structure_factors > 1e-6]
            structure_factors = structure_factors[structure_factors > 1e-6]
        else:
            # If no structure factors provided, use uniform intensity (all ones)
            structure_factors = torch.ones(miller_indices.shape[0])
        
        # Get scattering vectors and scattering angle.
        h = torch.einsum('ij,hj->hi', B, miller_indices)
        p_vectors = torch.einsum('ghi,gij,kj->gkh', torch.eye(3)[None,:,:] - self.strains[grains_hit], self.orientaions[grains_hit], h)
        p_vectors_norm = torch.linalg.norm(p_vectors, axis=-1)
        theta_angle = torch.asin( p_vectors_norm * wavelength / 4 / np.pi )

        # Filter out reflections far from the bragg-condition
        dp = torch.einsum('i,ghi->gh', xray_propagation_direction, p_vectors) / p_vectors_norm 
        does_diffract = torch.abs( dp + torch.sin(theta_angle) ) \
            < 3 * (self.max_misorientation + torch.linalg.norm(sample_rotation_during_exposure)) #IDEA: Consider a per-gaussian max misorientation
        grain_does_diffract, hkl_does_diffract = torch.where(does_diffract)

        if not torch.any(does_diffract):
            return torch.zeros(detector.shape)

        # Select the relevant reflections and flatten the grain- and symetry-indexes.
        misori_concentration_tensors = smeared_misorientation_tensors[grain_does_diffract]
        p_vectors = p_vectors[does_diffract]
        shape_concentration_tensors = intersection_shape_concentration_tensors[grain_does_diffract]

        if timing:
            print(f'Bragg-condition filterin took {time.time()-t0}. ({does_diffract.shape[0]*does_diffract.shape[1]} -> {torch.sum(does_diffract)})')
            t0 = time.time()

        # Do pole-figure part of the calculation
        mean_scattering_directions, partialities, outgoing_beam_divergence_tensor = _get_diffraction_arcsegment(
            p_vectors,
            misori_concentration_tensors,
            xray_propagation_direction,
            wavelength,
        )

        if timing:
            print(f'Reciprocal space part took {time.time()-t0}')
            t0 = time.time()

        # Splat grain realspace shapes (Consider using the non-strained non-azimuthally shifted directions to simplify gradients later)
        detectorspace_grainshape_projections, projected_thicknes_scale_factors = self.splat_grainshapes(
            mean_scattering_directions,
            shape_concentration_tensors,
            W,
            pixellengths,
        )

        if timing:
            print(f'Realspace proj took {time.time()-t0}')
            t0 = time.time()

        # Ray-trace onto detector plane
        pos = intersection_pos[grain_does_diffract]
        ray_lengths = torch.einsum('xi,i->x', detector_origin[None, :] - pos, detector_norm) / torch.einsum('xi,i->x', mean_scattering_directions, detector_norm)
        point_of_detector_intersection = pos + ray_lengths[:,None] * mean_scattering_directions
        uv_coords = torch.einsum('xi,vi,v->xv',point_of_detector_intersection - detector_origin[None, :], W, 1/pixellengths)

        # # Do smearing due to angular divergence
        # azimuthal_spread_xyz = azim_directions * ray_lengths[:, None] * azim_widths[:, None]
        # azimuthal_direction_uv = torch.einsum('xi,ui->xu', azimuthal_spread_xyz, W) / pixellengths[None, :]\
        #     / (1 - torch.einsum('xi,ui->xu', mean_scattering_directions, W)**2) # factor accounts for a smearing effect when the scattered beam
        #                                                                         # direction is not normal to the detector. I should re-write to
        #                                                                         # tensor-expressions for future-proofing. 
        
        W_scaled = W * 1 / pixellengths[:, None]
        divergence_smearing_tensor = torch.einsum('ui,xij,vj->xuv',
            W_scaled, outgoing_beam_divergence_tensor * ray_lengths[:, None, None]**2, W_scaled)


        # print(torch.linalg.eig(outgoing_beam_divergence_concentration_tensor[torch.argmax(partialities)]))
        
        # azimuthal_smearing_tensor = torch.einsum('xu,xv->xuv',azimuthal_direction_uv, azimuthal_direction_uv)
        detspace_splat_concentration = torch.linalg.inv( torch.linalg.inv(detectorspace_grainshape_projections) + divergence_smearing_tensor)
        intensity_spread_out_factor = torch.sqrt( torch.linalg.det(detspace_splat_concentration) / torch.linalg.det(detectorspace_grainshape_projections) )

        # print(torch.linalg.eig(azimuthal_smearing_tensor[torch.argmax(partialities)]).eigenvalues)
        # print(azimuthal_smearing_tensor[torch.argmax(partialities)])
        # print(torch.linalg.inv(detectorspace_grainshape_projections)[torch.argmax(partialities)])

        # Collect all intensity modifying factors
        polarization_factors = _polarization(mean_scattering_directions, beam.polarization_vector)
        solid_angle_factor = torch.abs(torch.einsum('xi,i->x',mean_scattering_directions, detector_norm))
        scalefactors = structure_factors[hkl_does_diffract] * projected_thicknes_scale_factors * partialities * intensity_spread_out_factor\
            * beam_intensity_factors[grain_does_diffract]*polarization_factors*solid_angle_factor

        does_diffract = scalefactors > 1e-6 * torch.max(scalefactors) # Discard weak peaks. Depends one unit-convention!

        if timing:
            print(f'Raytracing took {time.time()-t0}')
            t0 = time.time()

        peaks_batch_size = 20000
        n_batches = torch.sum(does_diffract) // peaks_batch_size + 1
        image_stack = torch.zeros(n_batches, *detector.shape)

        for peaks_batch in range(n_batches):

            image_stack[peaks_batch] = detector.render_gaussian_splats(
                uv_coords[does_diffract][peaks_batch*peaks_batch_size:(peaks_batch+1)*peaks_batch_size],
                scalefactors[does_diffract][peaks_batch*peaks_batch_size:(peaks_batch+1)*peaks_batch_size],
                detspace_splat_concentration[does_diffract][peaks_batch*peaks_batch_size:(peaks_batch+1)*peaks_batch_size],
            )

        f = torch.sum(image_stack, axis=0)

        if timing:
            print(f'Rasterization took {time.time()-t0}')
            t0 = time.time()

        return f
    

    def splat_grainshapes(
        self,
        mean_scattering_directions: Tensor,
        shape_concentration_tensors: Tensor,
        W: Tensor,
        pixellengths: Tensor,
    ):
        """ Project the laoratory space shape-concentration-tensors of a range of grains along a the scattering directions
        into 2D detector pixels space.

        Parameters
        ----------
        mean_scattering_directions : Tensor
            Scattering direction unit vectors, shape ``(N, 3)``
        shape_concentration_tensors : Tensor
            Shape concentration tensors in laboratory coordinates, shape ``(N, 3, 3)``
        W : Tensor
            Pixel-direction unit vectors stacked, shape ``(2, 3)``
        pixellengths : Tensor
            Pixel lengths, shape ``(2, 3)``
            
        Returns
        -------
        projected_shape_pixelunits : Tensor
            Concentarion tensor of the projected grainshape in detector pixel units, shape ``(N, 2, 2)``  
        projected_thicknes_scale_factors : Tensor
            Intensity scaling factor due the projected thickness of the grain, shape ``(N,)``
        """

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


    def transform(
            self,
            rigid_body_motion : RigidBodyMotion,
            time : float = 1.0,
        ):
        """Transform the polycrystal by performing a rigid body motion.

        This updates all the sample-information in-place.

        Parameters
        ----------
        rigid_body_motion : RigidBodyMotion
            Rigid body motion object describing the polycrystal transformation
            as a function of time on the domain ``time=[0, 1]``.
        time : float
            Time between ``[0, 1]`` at which to call the rigid body motion.
        """

        # Get rotation matrix and translation vector.
        Rot_mat = rigid_body_motion.rotator.get_rotation_matrix(
            rigid_body_motion.rotation_angle * time
        )
        translation_vector = rigid_body_motion.translation * time

        # Rotate vectors:
        self.positions = torch.einsum('ij,gj->gi', Rot_mat, self.positions-rigid_body_motion.origin[None,:])+rigid_body_motion.origin[None,:]

        # Rotate compose rotations
        self.orientaions = torch.einsum('ij,gjk->gik', Rot_mat, self.orientaions)

        #Rotate tensors
        self.shape_concentration_tensors = torch.einsum('ij,gjk,lk ->gil', Rot_mat, self.shape_concentration_tensors, Rot_mat)
        self.misori_concentration_tensors = torch.einsum('ij,gjk,lk ->gil', Rot_mat, self.misori_concentration_tensors, Rot_mat)
        self.strains = torch.einsum('ij,gjk,lk ->gil', Rot_mat, self.strains, Rot_mat)
        
        #Translate
        self.positions = self.positions + translation_vector[None, :] 


    # ------------------------------------------------------------------------------------------
    # The methods below here are for computing polefigures, not needed for diffraction patterns.
    # ------------------------------------------------------------------------------------------
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


        # A = form_a_mat(self.phase.unit_cell)
        # B = 2 * np.pi * np.linalg.inv(A).T
        B = form_b_mat(self.phase.unit_cell)
        h = torch.tensor(B @ hkl)
        h = h / torch.linalg.norm(h)

        levi_cita_symbol = np.zeros((3,3,3))
        levi_cita_symbol[0, 1, 2] = 1
        levi_cita_symbol[1, 2, 0] = 1
        levi_cita_symbol[2, 0, 1] = 1
        levi_cita_symbol[0, 2, 1] = -1
        levi_cita_symbol[1, 0, 2] = -1
        levi_cita_symbol[2, 1, 0] = -1
        levi_cita_symbol = torch.tensor(levi_cita_symbol)

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
