"""The polycrystal module is used to represent a polycrystalline sample. The :class:`xrd_simulator.polycrystal.Polycrystal`
object holds the function :func:`xrd_simulator.polycrystal.Polycrystal.diffract` which may be used to compute diffraction.
To move the sample spatially, the function :func:`xrd_simulator.polycrystal.Polycrystal.transform` can be used.
Here is a minimal example of how to instantiate a polycrystal object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_polycrystal.py

Below follows a detailed description of the polycrystal class attributes and functions.

"""

import copy
from multiprocessing import Pool
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import dill
from xfab import tools
from xrd_simulator.scattering_unit import ScatteringUnit
from xrd_simulator import utils, laue
from xrd_simulator.cuda import frame,pd

if frame != np:
    frame.array = frame.tensor

def _diffract(dict):
    """
    Compute diffraction for a subset of the polycrystal.

    Args:
        args (dict): A dictionary containing the following keys:
            - 'beam' (Beam): Object representing the incident X-ray beam.
            - 'detector' (Detector): Object representing the X-ray detector.
            - 'rigid_body_motion' (RigidBodyMotion): Object describing the polycrystal's transformation.
            - 'phases' (list): List of Phase objects representing the phases present in the polycrystal.
            - 'espherecentroids' (numpy.ndarray): Array containing the centroids of the scattering elements.
            - 'eradius' (numpy.ndarray): Array containing the radii of the scattering elements.
            - 'orientation_lab' (numpy.ndarray): Array containing orientation matrices in laboratory coordinates.
            - 'eB' (numpy.ndarray): Array containing per-element 3x3 B matrices mapping hkl values to crystal coordinates.
            - 'element_phase_map' (numpy.ndarray): Array mapping elements to phases.
            - 'ecoord' (numpy.ndarray): Array containing coordinates of the scattering elements.
            - 'verbose' (bool): Flag indicating whether to print progress.
            - 'proximity' (bool): Flag indicating whether to remove grains unlikely to be hit by the beam.
            - 'BB_intersection' (bool): Flag indicating whether to use Bounding-Box intersection for speed.

    Returns:
        list: A list of ScatteringUnit objects representing diffraction events.
    """

    beam = dict["beam"]
    detector = dict["detector"]
    rigid_body_motion = dict["rigid_body_motion"]
    phases = dict["phases"]
    espherecentroids = frame.array(dict["espherecentroids"])
    orientation_lab = dict["orientation_lab"]
    eB = dict["eB"]
    element_phase_map = frame.array(dict["element_phase_map"])

    rho_0_factor = frame.matmul(-beam.wave_vector,rigid_body_motion.rotator.K2)
    rho_1_factor = frame.matmul(beam.wave_vector,rigid_body_motion.rotator.K)
    rho_2_factor = frame.matmul(beam.wave_vector,(frame.eye(3, 3) + rigid_body_motion.rotator.K2))

    peaks_df = frame.empty((0,10),dtype=frame.float32)  # We create a dataframe to store all the relevant values for each individual reflection inr an organized manner

    # For each phase of the sample, we compute all reflections at once in a vectorized manner
    for i, phase in enumerate(phases):

        # Get all scatterers belonging to one phase at a time, and the corresponding miller indices.
        grain_indices = frame.where(element_phase_map == i)[0]
        miller_indices = frame.array(phase.miller_indices, dtype=frame.float32)

        # Retrieve the structure factors of the miller indices for this phase, exclude the miller incides with zero structure factor
        structure_factors = frame.sum(frame.array(phase.structure_factors, dtype=frame.float32)**2,axis=1)
        miller_indices = miller_indices[structure_factors>1e-6]
        structure_factors = structure_factors[structure_factors>1e-6]

        # Get all scattering vectors for all scatterers in a given phase
        G_0 = laue.get_G(orientation_lab[grain_indices], eB[grain_indices], miller_indices)

        # Now G_0 and rho_factors are sent before computation to save memory when diffracting many grains.
        '''CHECK THIS FUNCTION BECAUSE THERE IS A BIG UGLY BUGG SOMEWHERE 2024-10-07'''
        grains, planes, times, G0_xyz =laue.find_solutions_to_tangens_half_angle_equation( 
                G_0,
                rho_0_factor,
                rho_1_factor,
                rho_2_factor,
                rigid_body_motion.rotation_angle,
            )
        # We now assemble the dataframes with the valid reflections for each grain and phase including time, hkl plane and G vector
        #Column names of peaks are 'grain_index','phase_number','h','k','l','structure_factors','times','G0_x','G0_y','G0_z')
        if frame is np:
            structure_factors = structure_factors[planes][:, frame.newaxis]  # Unsqueeze at axis 1
            grain_indices = grain_indices[grains][:, frame.newaxis]          # Unsqueeze at axis 1
            miller_indices = miller_indices[planes]
            phase_index = frame.full((G0_xyz.shape[0],), i)[:, frame.newaxis] 
            times = times[:,frame.newaxis]
            peaks = frame.concatenate((grain_indices, phase_index, miller_indices[:, frame.newaxis].squeeze(), structure_factors, times[:, frame.newaxis].squeeze(2), G0_xyz), axis=1)
            peaks_df = frame.concatenate((peaks_df, peaks), axis=0)
        else:
            structure_factors = structure_factors[planes].unsqueeze(1)
            grain_indices = grain_indices[grains].unsqueeze(1)
            miller_indices = miller_indices[planes]
            phase_index = frame.full((G0_xyz.shape[0],),i).unsqueeze(1)
            times = times.unsqueeze(1)
            peaks = frame.cat((grain_indices,phase_index,miller_indices,structure_factors,times,G0_xyz),dim=1)
            peaks_df = frame.cat([peaks_df, peaks], axis=0)
        del peaks

    Gxyz = rigid_body_motion.rotate(peaks_df[:,7:], -peaks_df[:,6]) #I dont know why the - sign is necessary, there is a bug somewhere and this is a patch. Sue me.
    K_out_xyz = (Gxyz + beam.wave_vector)

    if frame is np:
        Sources_xyz = rigid_body_motion(espherecentroids[peaks_df[:,0].astype(int)],peaks_df[:,6].astype(int))    
    else:
        Sources_xyz = rigid_body_motion(espherecentroids[peaks_df[:,0].int()],peaks_df[:,6].int())
    zd_yd_angle = detector.get_intersection(K_out_xyz,Sources_xyz)
    # Concatenate new columns
    if frame is np:
        peaks_df = frame.concatenate((peaks_df,Gxyz,K_out_xyz,Sources_xyz,zd_yd_angle),axis=1)
    else:
        peaks_df = frame.cat((peaks_df,Gxyz,K_out_xyz,Sources_xyz,zd_yd_angle),dim=1)
    """
        Column names of peaks_df are now
        0: 'grain_index'        10: 'Gx'        20: 'yd'
        1: 'phase_number'       11: 'Gy'        21: 'Incident_angle'
        2: 'h'                  12: 'Gz'
        3: 'k'                  13: 'K_out_x'
        4: 'l'                  14: 'K_out_y'
        5: 'structure_factors'  15: 'K_out_z'
        6: 'diffraction_times'  16: 'Source_x'
        7: 'G0_x'               17: 'Source_y'      
        8: 'G0_y'               18: 'Source_z'
        9: 'G0_z'               19: 'zd'           
    """

    # Filter out peaks not hitting the detector
    peaks_df = peaks_df[detector.contains(peaks_df[:,19], peaks_df[:,20])]

    # Filter out tets not illuminated
    peaks_df = peaks_df[peaks_df[:,17] < (beam.vertices[:, 1].max())]  
    peaks_df = peaks_df[peaks_df[:,17] > (beam.vertices[:, 1].min())]  
    peaks_df = peaks_df[peaks_df[:,18] < (beam.vertices[:, 2].max())]  
    peaks_df = peaks_df[peaks_df[:,18] > (beam.vertices[:, 2].min())]  

    return peaks_df


class Polycrystal:
    """Represents a multi-phase polycrystal as a tetrahedral mesh where each element can be a single crystal

    The polycrystal is created in laboratory coordinates. At instantiation it is assumed that the sample and
    lab coordinate systems are aligned.

    Args:
        mesh (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample. (At instantiation it is assumed that the sample and lab coordinate systems
            are aligned.)
        orientation (:obj:`numpy array`): Per element orientation matrices (sometimes known by the capital letter U),
            (``shape=(N,3,3)``) or (``shape=(3,3)``) if the orientation is the same for all elements. The orientation
            matrix maps from crystal coordinates to sample coordinates.
        strain (:obj:`numpy array`): Per element (Green-Lagrange) strain tensor, in lab coordinates, (``shape=(N,3,3)``)
            or (``shape=(3,3)``) if the strain is the same for all elements elements.
        phases (:obj:`xrd_simulator.phase.Phase` or :obj:`list` of :obj:`xrd_simulator.phase.Phase`): Phase of the
            polycrystal, or for multiphase samples, a list of all phases present in the polycrystal.
        element_phase_map (:obj:`numpy array`): Index of phase that elements belong to such that phases[element_phase_map[i]]
            gives the xrd_simulator.phase.Phase object of element number i. None if the sample is composed of a single phase.
            (Defaults to None)

    Attributes:
        mesh_lab (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a fixed lab frame coordinate system. This quantity is updated when the sample transforms.
        mesh_sample (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a sample metadata={'angle':angle,
                               'angles':n_angles,
                               'steps':n_scans,
                               'beam':beam_file,
                               'sample':polycrystal_file,
                               'detector':detector_file},coordinate system. This quantity is not updated when the sample transforms.
        orientation_lab (:obj:`numpy array`): Per element orientation matrices mapping from the crystal to the lab coordinate
            system, this quantity is updated when the sample transforms. (``shape=(N,3,3)``).
        orientation_sample (:obj:`numpy array`): Per element orientation matrices mapping from the crystal to the sample
            coordinate system.,  this quantity is not updated when the sample transforms. (``shape=(N,3,3)``).
        strain_lab (:obj:`numpy array`): Per element (Green-Lagrange) strain tensor in a fixed lab frame coordinate
            system, this quantity is updated when the sample transforms. (``shape=(N,3,3)``).
        strain_sample (:obj:`numpy array`): Per element (Green-Lagrange) strain tensor in a sample coordinate
            system., this quantity is not updated when the sample transforms. (``shape=(N,3,3)``).
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.
        element_phase_map (:obj:`numpy array`): Index of phase that elements belong to such that phases[element_phase_map[i]]
            gives the xrd_simulator.phase.Phase object of element number i.

    """

    def __init__(self, mesh, orientation, strain, phases, element_phase_map=None):

        self.orientation_lab = self._instantiate_orientation(orientation, mesh)
        self.strain_lab = self._instantiate_strain(strain, mesh)
        self.element_phase_map, self.phases = self._instantiate_phase(
            phases, element_phase_map, mesh
        )
        self._eB = self._instantiate_eB(
            self.orientation_lab,
            self.strain_lab,
            self.phases,
            self.element_phase_map,
            mesh,
        )

        # Assuming sample and lab frames to be aligned at instantiation.
        self.mesh_lab = copy.deepcopy(mesh)
        self.mesh_sample = copy.deepcopy(mesh)
        self.strain_sample = np.copy(self.strain_lab)
        self.orientation_sample = np.copy(self.orientation_lab)

    def diffract(
        self,
        beam,
        detector,
        rigid_body_motion,
        min_bragg_angle=0,
        max_bragg_angle=None,
        number_of_frames=1,
    ):
        """Compute diffraction from the rotating and translating polycrystal while illuminated by an xray beam.

        The xray beam interacts with the polycrystal producing scattering units which are stored in a detector frame.
        The scattering units may be rendered as pixelated patterns on the detector by using a detector rendering
        option.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of xrays.
            detector (:obj:`xrd_simulator.detector.Detector`): Object representing a flat rectangular detector.
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain (time=[0,1]) over which diffraction is to be
                computed.
            min_bragg_angle (:obj:`float`): Minimum Bragg angle (radians) below which to not compute diffraction.
                Defaults to 0.
            max_bragg_angle (:obj:`float`): Maximum Bragg angle (radians) after which to not compute diffraction. By default
                the max_bragg_angle is approximated by wrapping the detector corners in a cone with apex at the sample-beam
                intersection centroid for time=0 in the rigid_body_motion.
            verbose (:obj:`bool`): Prints progress. Defaults to True.
            number_of_processes (:obj:`int`): Optional keyword specifying the number of desired processes to use for diffraction
                computation. Defaults to 1, i.e a single processes.
            number_of_frames (:obj:`int`): Optional keyword specifying the number of desired temporally equidistantly spaced frames
                to be collected. Defaulrenderts to 1, which means that the detector reads diffraction during the full rigid body
                motion and integrates out the signal to a single frame. The number_of_frames keyword primarily allows for single
                rotation axis full 180 dgrs or 360 dgrs sample rotation data sets to be computed rapidly and convinently.
            proximity (:obj:`bool`): Set to False if all or most grains from the sample are expected to diffract.
                For instance, if the diffraction scan illuminates all grains from the sample at least once at a give angle/position.
            BB_intersection (:obj:`bool`): Set to True in order to assume the beam as a square prism, the scattering volume for the tetrahedra
                to be the whole tetrahedron and the scattering tetrahedra to be all those whose centroids are included in the square prism.
                Greatly speeds up computation, valid approximation for powder-like samples.

        """

        beam.wave_vector = frame.array(beam.wave_vector, dtype=frame.float32)

        min_bragg_angle, max_bragg_angle = self._get_bragg_angle_bounds(
            detector, beam, min_bragg_angle, max_bragg_angle
        )

        for phase in self.phases:
            phase.setup_diffracting_planes(beam.wavelength, min_bragg_angle, max_bragg_angle)

        args = {
                    "beam": beam,
                    "detector": detector,
                    "rigid_body_motion": rigid_body_motion,
                    "phases": self.phases,
                    "espherecentroids": self.mesh_lab.espherecentroids,
                    "orientation_lab": self.orientation_lab,
                    "eB": self._eB,
                    "element_phase_map": self.element_phase_map,
                }

        peaks_df = _diffract(args)

        if frame is np:
            bin_edges = frame.linspace(0, 1,number_of_frames + 1)
            frames = frame.digitize(peaks_df[:,6], bin_edges)
            frames = frames[:,frame.newaxis]
            peaks_df = frame.concatenate((peaks_df, frames), axis=1)      
        else:
            bin_edges = frame.linspace(0, 1, steps=number_of_frames + 1)
            frames = frame.bucketize(peaks_df[:,6].contiguous(), bin_edges).unsqueeze(1)
            peaks_df = frame.cat((peaks_df,frames),dim=1)
        return peaks_df

    def transform(self, rigid_body_motion, time):
        """Transform the polycrystal by performing a rigid body motion (translation + rotation)

                This function will update the polycrystal mesh (update in lab frame) with any dependent quantities,
                such as face normals etc. Likewise, it will update the per element crystal orientation
                matrices (U) as well as the lab frame description of strain tensors.
        tt`): Time between [0,1] at which to call the rigid body motion.

        """
        self.mesh_lab.update(rigid_body_motion, time)

        Rot_mat = rigid_body_motion.rotator.get_rotation_matrix(
            rigid_body_motion.rotation_angle * time
        )

        self.orientation_lab = np.matmul(Rot_mat, self.orientation_lab)

        self.strain_lab = np.matmul(np.matmul(Rot_mat, self.strain_lab), Rot_mat.T)

    def save(self, path, save_mesh_as_xdmf=True):
        """Save polycrystal to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.
            save_mesh_as_xdmf (:obj:`bool`): If true, saves the polycrystal mesh with associated
                strains and crystal orientations as a .xdmf for visualization (sample coordinates).
                The results can be vizualised with for instance paraview (https://www.paraview.org/).
                The resulting data fields of the mesh data are the 6 unique components of the strain
                tensor (in sample coordinates) and the 3 Bunge Euler angles (Bunge, H. J. (1982). Texture
                Analysis in Materials Science. London: Butterworths.). Additionally a single field specifying
                the material phases of the sample will be saved.

        """
        if not path.endswith(".pc"):
            pickle_path = path + ".pc"
            xdmf_path = path + ".xdmf"
        else:
            pickle_path = path
            xdmf_path = path.split(".")[0] + ".xdmf"
        with open(pickle_path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
        if save_mesh_as_xdmf:
            element_data = {}
            element_data["Strain Tensor Component xx"] = self.strain_sample[:, 0, 0]
            element_data["Strain Tensor Component yy"] = self.strain_sample[:, 1, 1]
            element_data["Strain Tensor Component zz"] = self.strain_sample[:, 2, 2]
            element_data["Strain Tensor Component xy"] = self.strain_sample[:, 0, 1]
            element_data["Strain Tensor Component xz"] = self.strain_sample[:, 0, 2]
            element_data["Strain Tensor Component yz"] = self.strain_sample[:, 1, 2]
            element_data["Bunge Euler Angle phi_1 [degrees]"] = []
            element_data["Bunge Euler Angle Phi [degrees]"] = []
            element_data["Bunge Euler Angle phi_2 [degrees]"] = []
            element_data["Misorientation from mean orientation [degrees]"] = []

            misorientations = utils._get_misorientations(self.orientation_sample)

            for U, misorientation in zip(self.orientation_sample, misorientations):
                phi_1, PHI, phi_2 = tools.u_to_euler(U)
                element_data["Bunge Euler Angle phi_1 [degrees]"].append(
                    np.degrees(phi_1)
                )
                element_data["Bunge Euler Angle Phi [degrees]"].append(np.degrees(PHI))
                element_data["Bunge Euler Angle phi_2 [degrees]"].append(
                    np.degrees(phi_2)
                )

                element_data["Misorientation from mean orientation [degrees]"].append(
                    np.degrees(misorientation)
                )

            element_data["Material Phase Index"] = self.element_phase_map
            self.mesh_sample.save(xdmf_path, element_data=element_data)

    @classmethod
    def load(cls, path):
        """Load polycrystal from disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        if not path.endswith(".pc"):
            raise ValueError("The loaded polycrystal file must end with .pc")
        with open(path, "rb") as f:
            loaded = dill.load(f)
            if frame is np:
                pass
            else:
                loaded.orientation_lab = frame.array(loaded.orientation_lab, dtype=frame.float32)
                loaded.strain_lab = frame.array(loaded.strain_lab, dtype=frame.float32)
                loaded.element_phase_map = frame.array(loaded.element_phase_map, dtype=frame.float32)
                loaded._eB = frame.array(loaded._eB, dtype=frame.float32)
               # loaded.mesh_lab = frame.array(loaded.mesh_lab, dtype=frame.float32)
             # loaded.mesh_sample = frame.array(loaded.mesh_sample, dtype=frame.float32)
                loaded.strain_sample = frame.array(loaded.strain_sample, dtype=frame.float32)
                loaded.orientation_sample = frame.array(loaded.orientation_sample, dtype=frame.float32)
            return loaded

    def _instantiate_orientation(self, orientation, mesh):
        """Instantiate the orientations using for smart multi shape handling."""
        if orientation.shape == (3, 3):
            orientation_lab = np.repeat(
                orientation.reshape(1, 3, 3), mesh.number_of_elements, axis=0
            )
        elif orientation.shape == (mesh.number_of_elements, 3, 3):
            orientation_lab = np.copy(orientation)
        else:
            raise ValueError("orientation input is of incompatible shape")
        return orientation_lab

    def _instantiate_strain(self, strain, mesh):
        """Instantiate the strain using for smart multi shape handling."""
        if strain.shape == (3, 3):
            strain_lab = np.repeat(
                strain.reshape(1, 3, 3), mesh.number_of_elements, axis=0
            )
        elif strain.shape == (mesh.number_of_elements, 3, 3):
            strain_lab = np.copy(strain)
        else:
            raise ValueError("strain input is of incompatible shape")
        return strain_lab

    def _instantiate_phase(self, phases, element_phase_map, mesh):
        """Instantiate the phases using for smart multi shape handling."""
        if not isinstance(phases, list):
            phases = [phases]
        if element_phase_map is None:
            if len(phases) > 1:
                raise ValueError("element_phase_map not set for multiphase polycrystal")
            element_phase_map = np.zeros((mesh.number_of_elements,), dtype=int)
        return element_phase_map, phases

    def _instantiate_eB(
        self, orientation_lab, strain_lab, phases, element_phase_map, mesh
    ):
        """Compute per element 3x3 B matrices that map hkl (Miller) values to crystal coordinates.

        (These are upper triangular matrices such that
            G_s = U * B G_hkl
        where G_hkl = [h,k,l] lattice plane miller indices and G_s is the sample frame diffraction vectors.
        and U are the crystal element orientation matrices.)

        """
        _eB = np.zeros((mesh.number_of_elements, 3, 3))
        B0s = np.zeros((len(phases), 3, 3))
        for i, phase in enumerate(phases):
            B0s[i] = tools.form_b_mat(phase.unit_cell)
            grain_indices = np.where(np.array(element_phase_map) == i)[0]
            _eB[grain_indices] = utils.lab_strain_to_B_matrix(
                strain_lab[grain_indices], orientation_lab[grain_indices], B0s[i]
            )

        return _eB

    def _get_bragg_angle_bounds(self, detector, beam, min_bragg_angle, max_bragg_angle):
        """Compute a maximum Bragg angle cut of based on the beam sample interection region centroid and detector corners.

        If the beam graces or misses the sample, the sample centroid is used.
        """
        if max_bragg_angle is None:
            mesh_nodes_contained_by_beam = self.mesh_lab.coord[beam.contains(self.mesh_lab.coord.T), :]
            mesh_nodes_contained_by_beam = frame.array(mesh_nodes_contained_by_beam, dtype=frame.float32)
            if mesh_nodes_contained_by_beam.shape[0] != 0:
                source_point = frame.mean(mesh_nodes_contained_by_beam, axis=0)
            else:
                source_point = frame.array(self.mesh_lab.centroid, dtype=frame.float32)
            max_bragg_angle = detector.get_wrapping_cone(beam.wave_vector, source_point).item()

        assert (
            min_bragg_angle >= 0
        ), "min_bragg_angle must be greater or equal than zero"
        assert max_bragg_angle > min_bragg_angle, (
            "max_bragg_angle "
            + str(np.degrees(max_bragg_angle))
            + "dgrs must be greater than min_bragg_angle "
            + str(np.degrees(min_bragg_angle))
            + "dgrs"
        )
        return min_bragg_angle, max_bragg_angle
