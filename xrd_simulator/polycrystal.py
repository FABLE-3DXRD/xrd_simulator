"""The polycrystal module is used to represent a polycrystalline sample. The :class:`xrd_simulator.polycrystal.Polycrystal`
object holds the function :func:`xrd_simulator.polycrystal.Polycrystal.diffract` which may be used to compute diffraction.
To move the sample spatially, the function :func:`xrd_simulator.polycrystal.Polycrystal.transform` can be used.
Here is a minimal example of how to instantiate a polycrystal object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_polycrystal.py

Below follows a detailed description of the polycrystal class attributes and functions.

"""

import copy
import xrd_simulator.cuda
import warnings
import numpy as np
import dill
from xfab import tools
from xrd_simulator import utils, laue
from xrd_simulator.scattering_factors import lorentz, polarization
import torch
import matplotlib.pyplot as plt
from xrd_simulator.utils import ensure_torch, ensure_numpy
from xrd_simulator.scattering_unit import ScatteringUnit

torch.set_default_dtype(torch.float64)


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

        orientation = ensure_torch(orientation)
        strain = ensure_torch(strain)

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
        rigid_body_motion,
        min_bragg_angle=0,
        max_bragg_angle=90
        * np.pi
        / 180,  # Can go higher, but this is a reasonable default
        powder=False,
        detector=None,
    ):
        """Compute diffraction from the rotating and translating polycrystal while illuminated by an xray beam.

        The xray beam interacts with the polycrystal producing scattering units which are stored in a detector torch.
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
                motion and integrates out the signal to a single torch. The number_of_frames keyword primarily allows for single
                rotation axis full 180 dgrs or 360 dgrs sample rotation data sets to be computed rapidly and convinently.
            proximity (:obj:`bool`): Set to False if all or most grains from the sample are expected to diffract.
                For instance, if the diffraction scan illuminates all grains from the sample at least once at a give angle/position.
            BB_intersection (:obj:`bool`): Set to True in order to assume the beam as a square prism, the scattering volume for the tetrahedra
                to be the whole tetrahedron and the scattering tetrahedra to be all those whose centroids are included in the square prism.
                Greatly speeds up computation, valid approximation for powder-like samples.

        """

        # beam.wave_vector = ensure_torch(beam.wave_vector)
        if detector is not None:
            min_bragg_angle, max_bragg_angle = self._get_bragg_angle_bounds(
                detector, beam, min_bragg_angle, max_bragg_angle
            )

        for phase in self.phases:
            phase.setup_diffracting_planes(
                beam.wavelength, min_bragg_angle, max_bragg_angle
            )

        peaks = self.compute_peaks(beam, rigid_body_motion)

        if powder is True:
            # Filter out tets not illuminated
            peaks = peaks[peaks[:, 17] < (beam.vertices[:, 1].max())]
            peaks = peaks[peaks[:, 17] > (beam.vertices[:, 1].min())]
            peaks = peaks[peaks[:, 18] < (beam.vertices[:, 2].max())]
            peaks = peaks[peaks[:, 18] > (beam.vertices[:, 2].min())]
            scattering_units = []
        else:
            peaks, scattering_units = self.compute_scattering_units(
                beam, rigid_body_motion, peaks
            )

        """ Column names labeled like:
            0: 'grain_index'        10: 'Gx'        20: 'polarization_factors'
            1: 'phase_number'       11: 'Gy'        
            2: 'h'                  12: 'Gz'        
            3: 'k'                  13: 'K_out_x'   
            4: 'l'                  14: 'K_out_y'   
            5: 'structure_factors'  15: 'K_out_z'
            6: 'diffraction_times'  16: 'Source_x'
            7: 'G0_x'               17: 'Source_y'      
            8: 'G0_y'               18: 'Source_z'
            9: 'G0_z'               19: 'lorentz_factors'           
        """
        column_names = [
            "grain_index",
            "phase_number",
            "h",
            "k",
            "l",
            "structure_factors",
            "diffraction_times",
            "G0_x",
            "G0_y",
            "G0_z",
            "Gx",
            "Gy",
            "Gz",
            "K_out_x",
            "K_out_y",
            "K_out_z",
            "Source_x",
            "Source_y",
            "Source_z",
            "lorentz_factors",
            "polarization_factors",
        ]

        # Wrap the peaks columns and scattering units into a dict to preserve information
        peaks_dict = {
            "peaks": peaks,
            "columns": column_names,
            "scattering_units": scattering_units,
        }

        return peaks_dict

    def compute_peaks(self, beam, rigid_body_motion):
        """
        Compute diffraction for a subset of the polycrystal.

                - 'beam' (Beam): Object representing the incident X-ray beam.
                - 'rigid_body_motion' (RigidBodyMotion): Object describing the polycrystal's transformation.

        Returns:
            list: A list of ScatteringUnit objects representing diffraction events.
        """

        beam = beam
        rigid_body_motion = rigid_body_motion
        phases = self.phases
        espherecentroids = self.mesh_lab.espherecentroids
        orientation_lab = self.orientation_lab
        eB = self._eB
        element_phase_map = self.element_phase_map

        rho_0_factor = torch.matmul(-beam.wave_vector, rigid_body_motion.rotator.K2)
        rho_1_factor = torch.matmul(beam.wave_vector, rigid_body_motion.rotator.K)
        rho_2_factor = torch.matmul(
            beam.wave_vector, (torch.eye(3, 3) + rigid_body_motion.rotator.K2)
        )

        peaks = torch.empty(
            (0, 10)
        )  # We create a dataframe to store all the relevant values for each individual reflection inr an organized manner

        # For each phase of the sample, we compute all reflections at once in a vectorized manner
        for i, phase in enumerate(phases):

            # Get all scatterers belonging to one phase at a time, and the corresponding miller indices.
            grain_indices = torch.where(element_phase_map == i)[0]
            miller_indices = ensure_torch(phase.miller_indices)

            # # Retrieve the structure factors of the miller indices for this phase, exclude the miller incides with zero structure factor
            structure_factors = torch.sqrt(
                torch.sum(ensure_torch(phase.structure_factors) ** 2, axis=1)
            )
            miller_indices = miller_indices[structure_factors > 1e-6]
            structure_factors = structure_factors[structure_factors > 1e-6]

            # Get all scattering vectors for all scatterers in a given phase
            G_0 = laue.get_G(
                orientation_lab[grain_indices], eB[grain_indices], miller_indices
            )

            # Now G_0 and rho_factors are sent before computation to save memory when diffracting many grains.
            grains, planes, times, G0_xyz = (
                laue.find_solutions_to_tangens_half_angle_equation(
                    G_0,
                    rho_0_factor,
                    rho_1_factor,
                    rho_2_factor,
                    rigid_body_motion.rotation_angle,
                )
            )

            # We now assemble the tensors with the valid reflections for each grain and phase including time, hkl plane and G vector
            # Column names of peaks are 'grain_index','phase_number','h','k','l','structure_factors','times','G0_x','G0_y','G0_z')
            del G_0

            structure_factors = structure_factors[planes].unsqueeze(1)
            grain_indices = grain_indices[grains].unsqueeze(1)
            miller_indices = miller_indices[planes]
            phase_index = torch.full((G0_xyz.shape[0],), i).unsqueeze(1)
            times = times.unsqueeze(1)
            peaks_ith_phase = torch.cat(
                (
                    grain_indices,
                    phase_index,
                    miller_indices,
                    structure_factors,
                    times,
                    G0_xyz,
                ),
                dim=1,
            )
            peaks = torch.cat([peaks, peaks_ith_phase], axis=0)

        # Rotated G-vectors
        Gxyz = rigid_body_motion.rotate(peaks[:, 7:10], peaks[:, 6])

        # Outgoing scattering vectors
        K_out_xyz = Gxyz + beam.wave_vector

        # Lorentz factor
        lorentz_factors = lorentz(
            beam.wave_vector, K_out_xyz, rigid_body_motion.rotation_axis
        )
        # Polarization factor
        polarization_factors = polarization(K_out_xyz, beam.polarization_vector)

        Sources_xyz = rigid_body_motion(
            espherecentroids[peaks[:, 0].int()], peaks[:, 6]
        )
        peaks = torch.cat(
            (
                peaks,
                Gxyz,
                K_out_xyz,
                Sources_xyz,
                lorentz_factors.unsqueeze(1),
                polarization_factors.unsqueeze(1),
            ),
            dim=1,
        )

        """
            Column names of peaks are
            0: 'grain_index'        10: 'Gx'        20: 'polarization_factors'
            1: 'phase_number'       11: 'Gy'        
            2: 'h'                  12: 'Gz'        
            3: 'k'                  13: 'K_out_x'   
            4: 'l'                  14: 'K_out_y'   
            5: 'structure_factors'  15: 'K_out_z'
            6: 'diffraction_times'  16: 'Source_x'
            7: 'G0_x'               17: 'Source_y'      
            8: 'G0_y'               18: 'Source_z'
            9: 'G0_z'               19: 'lorentz_factors'           
        """

        return peaks

    def compute_scattering_units(self, beam, rigid_body_motion, peaks):
        peaks = ensure_numpy(peaks)
        beam = beam
        rigid_body_motion = rigid_body_motion
        phases = self.phases
        element_vertices_0 = self.mesh_lab.coord[
            self.mesh_lab.enod[peaks[:, 0].astype(int)]
        ]  # For each peak: tet x vertex x coordinate
        element_vertices = rigid_body_motion(
            element_vertices_0, peaks[:, 6]
        )  # vertices and times

        scattering_units = []
        filtered_peaks = []

        """Compute the true intersection of each tet with the beam to get the true scattering volume."""
        for ei in range(element_vertices.shape[0]):
            scattering_region = beam.intersect(element_vertices[ei])

            if scattering_region is not None:
                scattering_unit = ScatteringUnit(
                    scattering_region,
                    peaks[ei, 13:16],  # outgoing wavevector
                    beam.wave_vector,
                    beam.wavelength,
                    beam.polarization_vector,
                    rigid_body_motion.rotation_axis,
                    peaks[ei, 6],  # time
                    phases[peaks[ei, 1].astype(int)],  # phase
                    list(peaks[ei, 2:5].astype(int)),  # hkl index
                    ei,
                )
                filtered_peaks.append(peaks[ei, :])
                scattering_units.append(scattering_unit)
        filtered_peaks = np.vstack(filtered_peaks)
        return ensure_torch(filtered_peaks), scattering_units

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
        self.orientation_lab = torch.matmul(Rot_mat, self.orientation_lab)
        self.strain_lab = torch.matmul(
            torch.matmul(Rot_mat, self.strain_lab), Rot_mat.transpose(2, 1)
        )

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
            loaded.orientation_lab = ensure_torch(
                loaded.orientation_lab, dtype=torch.float64
            )
            loaded.strain_lab = ensure_torch(loaded.strain_lab)
            loaded.element_phase_map = ensure_torch(
                loaded.element_phase_map, dtype=torch.float64
            )
            loaded._eB = ensure_torch(loaded._eB)
            loaded.mesh_lab = cls._move_mesh_to_gpu(loaded.mesh_lab)
            loaded.mesh_sample = cls._move_mesh_to_gpu(loaded.mesh_sample)
            loaded.strain_sample = ensure_torch(
                loaded.strain_sample, dtype=torch.float64
            )
            loaded.orientation_sample = ensure_torch(
                loaded.orientation_sample, dtype=torch.float64
            )
            return loaded

    def _instantiate_orientation(self, orientation, mesh):
        """Instantiate the orientations using for smart multi shape handling."""
        if orientation.shape == (3, 3):
            orientation_lab = torch.repeat_interleave(
                ensure_torch(orientation).unsqueeze(0),
                mesh.number_of_elements,
                dim=0,
            )
        elif orientation.shape == (mesh.number_of_elements, 3, 3):
            orientation_lab = ensure_torch(orientation)
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
        else:
            element_phase_map = ensure_torch(element_phase_map)
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
            grain_indices = np.where(ensure_torch(element_phase_map) == i)[0]
            _eB[grain_indices] = utils.lab_strain_to_B_matrix(
                strain_lab[grain_indices], orientation_lab[grain_indices], B0s[i]
            )

        return _eB

    def _get_bragg_angle_bounds(self, detector, beam, min_bragg_angle, max_bragg_angle):
        """Compute a maximum Bragg angle cut of based on the beam sample interection region centroid and detector corners.

        If the beam graces or misses the sample, the sample centroid is used.
        """

        mesh_nodes_contained_by_beam = self.mesh_lab.coord[
            beam.contains(self.mesh_lab.coord.T), :
        ]
        mesh_nodes_contained_by_beam = ensure_torch(mesh_nodes_contained_by_beam)
        if mesh_nodes_contained_by_beam.shape[0] != 0:
            source_point = torch.mean(mesh_nodes_contained_by_beam, axis=0)
        else:
            source_point = ensure_torch(self.mesh_lab.centroid)
        max_bragg_angle = detector.get_wrapping_cone(
            beam.wave_vector, source_point
        ).item()

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

    def _move_mesh_to_gpu(mesh):
        mesh.coord = ensure_torch(mesh.coord)
        mesh.enod = ensure_torch(mesh.enod, dtype=torch.int32)
        mesh.dof = ensure_torch(mesh.dof)
        mesh.efaces = ensure_torch(mesh.efaces, dtype=torch.int32)
        mesh.enormals = ensure_torch(mesh.enormals)
        mesh.ecentroids = ensure_torch(mesh.ecentroids)
        mesh.eradius = ensure_torch(mesh.eradius)
        mesh.espherecentroids = ensure_torch(mesh.espherecentroids)
        mesh.centroid = ensure_torch(mesh.centroid)
        return mesh
