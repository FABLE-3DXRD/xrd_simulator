"""The polycrystal module is used to represent a polycrystalline sample. The :class:`xrd_simulator.polycrystal.Polycrystal`
object holds the function :func:`xrd_simulator.polycrystal.Polycrystal.diffract` which may be used to compute diffraction.
To move the sample spatially, the function :func:`xrd_simulator.polycrystal.Polycrystal.transform` can be used.
Here is a minimal example of how to instantiate a polycrystal object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_polycrystal.py

Below follows a detailed description of the polycrystal class attributes and functions.

"""
import numpy as np
import dill
import copy
from xfab import tools
from xrd_simulator.scattering_unit import ScatteringUnit
from xrd_simulator import utils, laue


class Polycrystal():

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
        strain (:obj:`numpy array`): Per element strain tensor, in lab coordinates, (``shape=(N,3,3)``) or (``shape=(3,3)``)
            if the strain is the same for all elements elements.
        phases (:obj:`xrd_simulator.phase.Phase` or :obj:`list` of :obj:`xrd_simulator.phase.Phase`): Phase of the
            polycrystal, or for multiphase samples, a list of all phases present in the polycrystal.
        element_phase_map (:obj:`numpy array`): Index of phase that elements belong to such that phases[element_phase_map[i]]
            gives the xrd_simulator.phase.Phase object of element number i. None if the sample is composed of a single phase.
            (Defaults to None)

    Attributes:
        mesh_lab (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a fixed lab frame coordinate system. This quantity is updated when the sample transforms.
        mesh_sample (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a sample coordinate system. This quantity is not updated when the sample transforms.
        orientation_lab (:obj:`numpy array`): Per element orientation matrices mapping from the crystal to the lab coordinate
            system, this quantity is updated when the sample transforms. (``shape=(N,3,3)``).
        orientation_sample (:obj:`numpy array`): Per element orientation matrices mapping from the crystal to the sample
            coordinate system.,  this quantity is not updated when the sample transforms. (``shape=(N,3,3)``).
        strain_lab (:obj:`numpy array`): Per element strain tensor in a fixed lab frame coordinate
            system, this quantity is updated when the sample transforms. (``shape=(N,3,3)``).
        strain_sample (:obj:`numpy array`): Per element strain tensor in a sample coordinate
            system., this quantity is not updated when the sample transforms. (``shape=(N,3,3)``).
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.
        element_phase_map (:obj:`numpy array`): Index of phase that elements belong to such that phases[element_phase_map[i]]
            gives the xrd_simulator.phase.Phase object of element number i.

    """

    def __init__(
            self,
            mesh,
            orientation,
            strain,
            phases,
            element_phase_map=None):

        self.orientation_lab = self._instantiate_orientation(orientation, mesh)
        self.strain_lab = self._instantiate_strain(strain, mesh)
        self.element_phase_map, self.phases = self._instantiate_phase(phases, element_phase_map, mesh)
        self._eB = self._instantiate_eB(self.orientation_lab, self.strain_lab, self.phases, self.element_phase_map, mesh)

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
            verbose=True):
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
                the max_bragg_angle is approximated by wrapping the detector corners in a cone with apex at the sample
                centroid.
            verbose (:obj:`bool`): Prints progress. Defaults to True.

        """

        min_bragg_angle, max_bragg_angle = self._get_bragg_angle_bounds(
            detector, beam, min_bragg_angle, max_bragg_angle)

        for phase in self.phases:
            with utils._verbose_manager(verbose):
                phase.setup_diffracting_planes(
                    beam.wavelength,
                    min_bragg_angle,
                    max_bragg_angle)

        rho_0_factor = -beam.wave_vector.dot(rigid_body_motion.rotator.K2)
        rho_1_factor = beam.wave_vector.dot(rigid_body_motion.rotator.K)
        rho_2_factor = beam.wave_vector.dot(
            np.eye(3, 3) + rigid_body_motion.rotator.K2)

        scattering_units = []

        proximity_intervals = beam._get_proximity_intervals(
            self.mesh_lab.espherecentroids, self.mesh_lab.eradius, rigid_body_motion)

        for ei in range(self.mesh_lab.number_of_elements):

            if verbose:
                progress_bar_message = "Computing scattering from a total of " + \
                    str(self.mesh_lab.number_of_elements) + " elements"
                progress_fraction = float(
                    ei + 1) / self.mesh_lab.number_of_elements
                utils._print_progress(
                    progress_fraction,
                    message=progress_bar_message)

            # skip elements not illuminated
            if proximity_intervals[ei][0] is None:
                continue

            element_vertices_0 = self.mesh_lab.coord[self.mesh_lab.enod[ei], :]
            G_0 = laue.get_G(self.orientation_lab[ei], self._eB[ei],
                             self.phases[self.element_phase_map[ei]].miller_indices.T)

            rho_0s = rho_0_factor.dot(G_0)
            rho_1s = rho_1_factor.dot(G_0)
            rho_2s = rho_2_factor.dot(G_0) + np.sum((G_0 * G_0), axis=0) / 2.

            for hkl_indx in range(G_0.shape[1]):
                for time in laue.find_solutions_to_tangens_half_angle_equation(
                        rho_0s[hkl_indx], rho_1s[hkl_indx], rho_2s[hkl_indx], rigid_body_motion.rotation_angle):
                    if time is not None:
                        if utils._contained_by_intervals(
                                time, proximity_intervals[ei]):
                            element_vertices = rigid_body_motion(
                                element_vertices_0.T, time).T

                            # TODO: Consider plane equations representation of
                            # elements avoiding ConvexHull calls in
                            # beam.intersect()
                            scattering_region = beam.intersect(
                                element_vertices)

                            if scattering_region is not None:
                                G = rigid_body_motion.rotate(
                                    G_0[:, hkl_indx], time)
                                scattered_wave_vector = G + beam.wave_vector
                                scattering_unit = ScatteringUnit(scattering_region,
                                                      scattered_wave_vector,
                                                      beam.wave_vector,
                                                      beam.wavelength,
                                                      beam.polarization_vector,
                                                      rigid_body_motion.rotation_axis,
                                                      time,
                                                      self.phases[self.element_phase_map[ei]],
                                                      hkl_indx)
                                scattering_units.append(scattering_unit)
        detector.frames.append(scattering_units)

    def transform(self, rigid_body_motion, time):
        """Transform the polycrystal by performing a rigid body motion (translation + rotation)

        This function will update the polycrystal mesh (update in lab frame) with any dependent quantities,
        such as face normals etc. Likewise, it will update the per element crystal orientation
        matrices (U) as well as the lab frame description of strain tensors.

        Args:
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].
            time (:obj:`float`): Time between [0,1] at which to call the rigid body motion.

        """
        angle_to_rotate = rigid_body_motion.rotation_angle * time
        Rot_mat = rigid_body_motion.rotator.get_rotation_matrix(
            angle_to_rotate)
        new_nodal_coordinates = rigid_body_motion(
            self.mesh_lab.coord.T, time=time).T
        self.mesh_lab.update(new_nodal_coordinates)
        for ei in range(self.mesh_lab.number_of_elements):
            self.orientation_lab[ei] = np.dot(
                Rot_mat, self.orientation_lab[ei])
            self.strain_lab[ei] = np.dot(
                Rot_mat, np.dot(
                    self.strain_lab[ei], Rot_mat.T))

    def save(self, path, save_mesh_as_xdmf=True):
        """Save polycrystal to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.
            save_mesh_as_xdmf (:obj:`bool`): If true, saves the polycrystal mesh with associated
                strains and crystal orientations as a .xdmf for visualization (sample coordinates).
                The results can be vizualised with for instance paraview (https://www.paraview.org/).
                The resulting data fields of the mesh data are the 6 unique components of the small strain 
                tensor (in sample coordinates) and the 3 Bunge Euler angles (Bunge, H. J. (1982). Texture
                Analysis in Materials Science. London: Butterworths.). Additionally a single field specifying
                the material phases of the sample will be saved.

        """
        if not path.endswith(".pc"):
            pickle_path = path + ".pc"
            xdmf_path = path + ".xdmf"
        else:
            pickle_path = path
            xdmf_path = path.split('.')[0]+ ".xdmf"
        with open(pickle_path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
        if save_mesh_as_xdmf:
            element_data = {}
            element_data['Small Strain Tensor Component xx'] = self.strain_sample[:, 0, 0]
            element_data['Small Strain Tensor Component yy'] = self.strain_sample[:, 1, 1]
            element_data['Small Strain Tensor Component zz'] = self.strain_sample[:, 2, 2]
            element_data['Small Strain Tensor Component xy'] = self.strain_sample[:, 0, 1]
            element_data['Small Strain Tensor Component xz'] = self.strain_sample[:, 0, 2]
            element_data['Small Strain Tensor Component yz'] = self.strain_sample[:, 1, 2]
            element_data['Bunge Euler Angle phi_1'] = []
            element_data['Bunge Euler Angle Phi'] = []
            element_data['Bunge Euler Angle phi_2'] = []
            for U in self.orientation_sample:
                phi_1, PHI, phi_2 = tools.u_to_euler(U)
                element_data['Bunge Euler Angle phi_1'].append(phi_1)
                element_data['Bunge Euler Angle Phi'].append(PHI)
                element_data['Bunge Euler Angle phi_2'].append(phi_2)
            element_data['Material Phase Index'] = self.element_phase_map
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
        with open(path, 'rb') as f:
            return dill.load(f)

    def _instantiate_orientation(self, orientation, mesh):
        """Instantiate the orientations using for smart multi shape handling.

        """
        if orientation.shape==(3,3):
            orientation_lab = np.repeat(
                orientation.reshape(1, 3, 3), mesh.number_of_elements, axis=0)
        elif orientation.shape==(mesh.number_of_elements,3,3):
            orientation_lab = np.copy(orientation)
        else:
            raise ValueError("orientation input is of incompatible shape")
        return orientation_lab

    def _instantiate_strain(self, strain, mesh):
        """Instantiate the strain using for smart multi shape handling.

        """
        if strain.shape==(3,3):
            strain_lab = np.repeat(strain.reshape(1, 3, 3), mesh.number_of_elements, axis=0)
        elif strain.shape==(mesh.number_of_elements,3,3):
            strain_lab = np.copy(strain)
        else:
            raise ValueError("strain input is of incompatible shape")
        return strain_lab

    def _instantiate_phase(self, phases, element_phase_map, mesh):
        """Instantiate the phases using for smart multi shape handling.

        """
        if not isinstance(phases, list):
            phases = [phases]
        if element_phase_map is None:
            if len(phases)>1:
                raise ValueError("element_phase_map not set for multiphase polycrystal")
            element_phase_map = np.zeros((mesh.number_of_elements,), dtype=int)
        return element_phase_map, phases

    def _instantiate_eB(self, orientation_lab, strain_lab, phases, element_phase_map, mesh):
        """Compute per element 3x3 B matrices that map hkl (Miller) values to crystal coordinates.

            (These are upper triangular matrices such that
                G_s = U * B G_hkl
            where G_hkl = [h,k,l] lattice plane miller indices and G_s is the sample frame diffraction vectors.
            and U are the crystal element orientation matrices.)

        """
        _eB = np.zeros((mesh.number_of_elements, 3, 3))
        for i in range(mesh.number_of_elements):
            _eB[i,:,:] = utils.lab_strain_to_B_matrix(strain_lab[i,:,:],
                                                      orientation_lab[i,:,:],
                                                      phases[element_phase_map[i]].unit_cell)
        return _eB

    def _get_bragg_angle_bounds(
            self,
            detector,
            beam,
            min_bragg_angle,
            max_bragg_angle):
        """Compute a maximum Bragg angle cut of based on the beam sample interection region centroid and detector corners.

        If the beam graces or misses the sample, the sample centroid is used.
        """
        if max_bragg_angle is None:
            mesh_nodes_contained_by_beam = np.array(
                [c for c in self.mesh_lab.coord if beam.contains(c)])
            if len(mesh_nodes_contained_by_beam) != 0:
                source_point = np.mean(
                    np.array([c for c in self.mesh_lab.coord if beam.contains(c)]), axis=0)
            else:
                source_point = self.mesh_lab.centroid
            max_bragg_angle = detector.get_wrapping_cone(
                beam.wave_vector, source_point)
        assert min_bragg_angle >= 0, "min_bragg_angle must be greater or equal than zero"
        assert max_bragg_angle > min_bragg_angle, "max_bragg_angle must be greater than min_bragg_angle"
        return min_bragg_angle, max_bragg_angle
