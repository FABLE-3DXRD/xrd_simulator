from scipy.spatial.transform import Rotation as R
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from xrd_simulator.phase import Phase
from xrd_simulator.utils import ensure_torch

from xfab.tools import form_a_mat


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
        grain_list: List[GaussianGrainish],
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
            max_angle=3*max_misorientation + (resolution_in_degrees*np.pi/180) * patch_size/2,
        )

        return f, polar, azim

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
