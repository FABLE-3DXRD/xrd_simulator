import numpy as np
from xrd_simulator.scatterer import Scatterer
from xrd_simulator import utils, laue
from xrd_simulator._pickleable_object import PickleableObject
import copy


class Polycrystal(PickleableObject):

    """Represents a multi-phase polycrystal as a tetrahedral mesh where each element can be a single crystal

    Args:
        mesh (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample. At instantiation it is assumed that the sample and lab coordinate systems
            are aligned.
        ephase (:obj:`numpy array`): Index of phase that elements belong to such that phases[ephase[i]] gives the
            xrd_simulator.phase.Phase object of element number i.
        eU (:obj:`numpy array`): Per element U (orinetation) matrices, (``shape=(N,3,3)``). At instantiation it
            is assumed that the sample and lab coordinate systems are aligned.
        eB (:obj:`numpy array`): Per element B (hkl to crystal mapper) matrices, (``shape=(N,3,3)``).
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.

    Attributes:
        mesh_lab (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a fixed lab frame coordinate system.
        mesh_sample (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the
            geometry of the sample in a sample coordinate system.
        ephase (:obj:`numpy array`): Index of phase that elements belong to such that phases[ephase[i]] gives the
            xrd_simulator.phase.Phase object of element number i.
        eU_lab (:obj:`numpy array`): Per element U (orientation) matrices in a fixed lab frame coordinate
            system, (``shape=(N,3,3)``).
        eU_sample (:obj:`numpy array`): Per element U (orientation) matrices in a sample coordinate
            system., (``shape=(N,3,3)``).
        eB (:obj:`numpy array`): Per element B (hkl to crystal mapper) matrices, (``shape=(N,3,3)``).
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.

    """

    def __init__(self, mesh, ephase, eU, eB, phases):

        assert eU.shape[0] == mesh.enod.shape[0], "Every crystal element must have an orientation."
        assert eB.shape[0] == mesh.enod.shape[0], "Every crystal element must have a deformation state."
        for i in range(1, 3):
            assert eU.shape[i] == 3
            assert eB.shape[i] == 3

        # TODO: Allow specifying strain rather than eB as eB is much more
        # non intuitive quantity.

        # Assuming sample and lab frames to be aligned at instantiation.
        self.mesh_lab = copy.deepcopy(mesh)
        self.eU_lab = eU.copy()
        self.mesh_sample = copy.deepcopy(mesh)
        self.eU_sample = eU.copy()

        self.phases = phases
        self.ephase = ephase
        self.eB = eB

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
                polycrystal transformation as a funciton of time on the domain time=[0,1].
            min_bragg_angle (:obj:`float`): Minimum Bragg angle (radians) below wich to not compute diffraction. Defaults to 0.
            max_bragg_angle (:obj:`float`): Minimum Bragg angle (radians) after wich to not compute diffraction. By default the
                max_bragg_angle is approximated by wrapping the detector corners in a cone with apex at the sample centroid.
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
            rigid_body_motion.rotator.I +
            rigid_body_motion.rotator.K2)

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

            # skipp elements not illuminated
            if proximity_intervals[ei][0] is None:
                continue

            element_vertices_0 = self.mesh_lab.coord[self.mesh_lab.enod[ei], :]
            G_0 = laue.get_G(
                self.eU_lab[ei], self.eB[ei], self.phases[self.ephase[ei]].miller_indices.T)

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
                                                      self.phases[self.ephase[ei]],
                                                      hkl_indx)
                                scatterers.append(scatterer)
        detector.frames.append(scatterers)

    def _get_bragg_angle_bounds(
            self,
            detector,
            beam,
            min_bragg_angle,
            max_bragg_angle):
        """Compute a maximum Bragg angle cutof based on the beam sample interection region centroid and detector corners.

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
        matrices (U) such that the updated matrix will

        Args:
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a funciton of time on the domain time=[0,1].
            time (:obj:`float`): Time between [0,1] at which to call the rigid body motion.


        """
        new_nodal_coordinates = rigid_body_motion(
            self.mesh_lab.coord.T, time=time).T
        self.mesh_lab.update(new_nodal_coordinates)
        for ei in range(self.mesh_lab.number_of_elements):
            self.eU_lab[ei] = rigid_body_motion.rotate(
                self.eU_lab[ei], time=time)
