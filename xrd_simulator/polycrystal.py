import numpy as np
from xrd_simulator.scatterer import Scatterer
from xrd_simulator import utils, laue
from xrd_simulator._pickleable_object import PickleableObject
import copy

from xrd_simulator.xfab import tools


class Polycrystal(PickleableObject):

    """Represents a multi-phase polycrystal as a tetrahedral mesh where each element can be a single crystal

    Args:
        mesh (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample. At instantiation it is assumed that the sample and lab coordinate systems
            are aligned.
        orientation (:obj:`numpy array`): Per element orientation matrices (sometimes known by the capital letter U),
            (``shape=(N,3,3)``) or (``shape=(3,3)``) if the orientation is uniform between elements. At instantiation it
            is assumed that the sample and lab coordinate systems are aligned.
        strain (:obj:`numpy array`): Per element strain tensor, (``shape=(N,3,3)``) or (``shape=(3,3)``) if the strain is
            uniform between elements.
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.
        element_phase_map (:obj:`numpy array`): Index of phase that elements belong to such that phases[element_phase_map[i]]
            gives the xrd_simulator.phase.Phase object of element number i. None if the sample is composed of a single phase.
            (Defaults to None)

    Attributes:
        mesh_lab (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a fixed lab frame coordinate system.
        mesh_sample (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a sample coordinate system.
        orientation_lab (:obj:`numpy array`): Per element orientation matrices in a fixed lab frame coordinate
            system, (``shape=(N,3,3)``). (sometimes known by the capital letter U)
        orientation_sample (:obj:`numpy array`): Per element orientation matrices in a sample coordinate
            system., (``shape=(N,3,3)``). (sometimes known by the capital letter U)
        strain_lab (:obj:`numpy array`): Per element strain tensor in a fixed lab frame coordinate
            system, (``shape=(N,3,3)``).
        strain_sample (:obj:`numpy array`): Per element strain tensor in a sample coordinate
            system., (``shape=(N,3,3)``).
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

        if len(orientation.shape) == 2:
            self.orientation_lab = np.repeat(
                orientation.reshape(
                    1, 3, 3), mesh.number_of_elements, axis=0)
        else:
            self.orientation_lab = np.copy(orientation)

        if len(strain.shape) == 2:
            self.strain_lab = np.repeat(strain.reshape(
                1, 3, 3), mesh.number_of_elements, axis=0)
        else:
            self.strain_lab = np.copy(strain)

        if not isinstance(phases, list):
            self.phases = [phases]
        else:
            self.phases = phases

        if len(self.phases) == 1:
            self.element_phase_map = np.zeros(
                (mesh.number_of_elements,), dtype=int)
        else:
            self.element_phase_map = element_phase_map

        self._eB = np.zeros((mesh.number_of_elements, 3, 3))
        for i in range(mesh.number_of_elements):
            self._eB[i,
                     :,
                     :] = utils.lab_strain_to_B_matrix(self.strain_lab[i,
                                                                       :,
                                                                       :],
                                                       self.orientation_lab[i,
                                                                            :,
                                                                            :],
                                                       self.phases[self.element_phase_map[i]].unit_cell)

        assert self.orientation_lab.shape[0] == mesh.number_of_elements, "Every crystal element must have an orientation."
        assert self._eB.shape[0] == mesh.number_of_elements, "Every crystal element must have a deformation state."
        for i in range(1, 3):
            assert orientation.shape[i] == 3
            assert self._eB.shape[i] == 3

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
        """Compute diffraction from the rotating and translating polycrystal sample while illuminated by an xray beam.

        The xray beam interacts with the polycrystal producing scattering regions which are stored in a detector frame.
        The scattering regions may be rendered as pixelated patterns on the detector by using a detector rendering
        option.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            detector (:obj:`xrd_simulator.detector.Detector`): Object representing a flat rectangular detector.
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a function of time on the domain time=[0,1].
            min_bragg_angle (:obj:`float`): Minimum Bragg angle (radians) below which to not compute diffraction.
                Defaults to 0.
            max_bragg_angle (:obj:`float`): Minimum Bragg angle (radians) after which to not compute diffraction. By default
                the max_bragg_angle is approximated by wrapping the detector corners in a cone with apex at the sample
                centroid.
            verbose (:obj:`bool`): Prints progress. Defaults to True.

        """

        min_bragg_angle, max_bragg_angle = self._get_bragg_angle_bounds(
            detector, beam, min_bragg_angle, max_bragg_angle)

        for phase in self.phases:
            phase.setup_diffracting_planes(
                beam.wavelength,
                min_bragg_angle,
                max_bragg_angle,
                verbose=verbose)

        c_0_factor = -beam.wave_vector.dot(rigid_body_motion.rotator.K2)
        c_1_factor = beam.wave_vector.dot(rigid_body_motion.rotator.K)
        c_2_factor = beam.wave_vector.dot(
            np.eye(3, 3) + rigid_body_motion.rotator.K2)

        scatterers = []

        proximity_intervals = beam.get_proximity_intervals(
            self.mesh_lab.espherecentroids, self.mesh_lab.eradius, rigid_body_motion)

        for ei in range(self.mesh_lab.number_of_elements):

            if verbose:
                progress_bar_message = "Computing scattering from a total of " + \
                    str(self.mesh_lab.number_of_elements) + " elements"
                progress_fraction = float(
                    ei + 1) / self.mesh_lab.number_of_elements
                utils.print_progress(
                    progress_fraction,
                    message=progress_bar_message)

            # skip elements not illuminated
            if proximity_intervals[ei][0] is None:
                continue

            element_vertices_0 = self.mesh_lab.coord[self.mesh_lab.enod[ei], :]
            G_0 = laue.get_G(self.orientation_lab[ei], self._eB[ei],
                             self.phases[self.element_phase_map[ei]].miller_indices.T)

            c_0s = c_0_factor.dot(G_0)
            c_1s = c_1_factor.dot(G_0)
            c_2s = c_2_factor.dot(G_0) + np.sum((G_0 * G_0), axis=0) / 2.

            for hkl_indx in range(G_0.shape[1]):
                for time in laue.find_solutions_to_tangens_half_angle_equation(
                        c_0s[hkl_indx], c_1s[hkl_indx], c_2s[hkl_indx], rigid_body_motion.rotation_angle):
                    if time is not None:
                        if utils.contained_by_intervals(
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
                                scatterer = Scatterer(scattering_region,
                                                      scattered_wave_vector,
                                                      beam.wave_vector,
                                                      beam.wavelength,
                                                      beam.polarization_vector,
                                                      rigid_body_motion.rotation_axis,
                                                      time,
                                                      self.phases[self.element_phase_map[ei]],
                                                      hkl_indx)
                                scatterers.append(scatterer)
        detector.frames.append(scatterers)

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

    def transform(self, rigid_body_motion, time):
        """Transform the polycrystal by performing a rigid body motion (translation + rotation)

        This function will update the polycrystal mesh (update in lab frame) with any dependent quantities,
        such as face normals etc. Likewise, it will update the per element crystallite orientation
        matrices (U) and the lab frame description of strain.

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
        """Save object to disc.

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.
            save_mesh_as_xdmf (:obj:`bool`): If true, saves the polycyrystal mesh with associated
                strains and crystal orientations as a .xdmf for visualization (sample coordinates).

        """
        super().save(path)
        if save_mesh_as_xdmf:
            element_data = {}
            element_data['$\\epsilon_{11}$'] = self.strain_sample[:, 0, 0]
            element_data['$\\epsilon_{22}$'] = self.strain_sample[:, 1, 1]
            element_data['$\\epsilon_{33}$'] = self.strain_sample[:, 2, 2]
            element_data['$\\epsilon_{12}$'] = self.strain_sample[:, 0, 1]
            element_data['$\\epsilon_{13}$'] = self.strain_sample[:, 0, 2]
            element_data['$\\epsilon_{23}$'] = self.strain_sample[:, 1, 2]
            element_data['$\\varphi_{1}$'] = []
            element_data['$\\Phi$'] = []
            element_data['$\\varphi_{2}$'] = []
            for U in self.orientation_sample:
                phi_1, PHI, phi_2 = tools.u_to_euler(U)
                element_data['$\\varphi_{1}$'].append(phi_1)
                element_data['$\\Phi$'].append(PHI)
                element_data['$\\varphi_{2}$'].append(phi_2)
            element_data['Phases'] = self.element_phase_map
            self.mesh_sample.save(path + ".xdmf", element_data=element_data)
