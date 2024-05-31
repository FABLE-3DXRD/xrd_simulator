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
import pandas as pd
import dill
from xfab import tools
from xrd_simulator.scattering_unit import ScatteringUnit
from xrd_simulator import utils, laue


def _diffract(dict):

    beam = dict["beam"]
    detector = dict["detector"]
    rigid_body_motion = dict["rigid_body_motion"]
    phases = dict["phases"]
    espherecentroids = dict["espherecentroids"]
    eradius = dict["eradius"]
    orientation_lab = dict["orientation_lab"]
    eB = dict["eB"]
    element_phase_map = dict["element_phase_map"]
    ecoord = dict["ecoord"]
    verbose = dict[
        "verbose"
    ]  # should be deprecated or repurposed since computation now takes place phase by phase not by individual scatterer
    proximity = dict["proximity"]
    BB_intersection = dict["BB_intersection"]
    number_of_elements = ecoord.shape[0]

    rho_0_factor = np.float32(-beam.wave_vector.dot(rigid_body_motion.rotator.K2))
    rho_1_factor = np.float32(beam.wave_vector.dot(rigid_body_motion.rotator.K))
    rho_2_factor = np.float32(
        beam.wave_vector.dot(np.eye(3, 3) + rigid_body_motion.rotator.K2)
    )

    if proximity:
        # Grains with no chance to be hit by the beam are removed beforehand, if proximity is toggled as True
        proximity_intervals = beam._get_proximity_intervals(
            espherecentroids, eradius, rigid_body_motion
        )
        possible_scatterers_mask = np.array(
            [pi[0] is not None for pi in proximity_intervals]
        )
        espherecentroids = np.float32(espherecentroids[possible_scatterers_mask])
        eradius = np.float32(eradius[possible_scatterers_mask])

        orientation_lab = np.float32(orientation_lab[possible_scatterers_mask])
        eB = np.float32(eB[possible_scatterers_mask])
        element_phase_map = element_phase_map[possible_scatterers_mask]
        ecoord = np.float32(ecoord[possible_scatterers_mask])

    reflections_df = (
        pd.DataFrame()
    )  # We create a dataframe to store all the relevant values for each individual reflection inr an organized manner
    scattering_units = []  # The output

    # For each phase of the sample, we compute all reflections at once in a vectorized manner
    for i, phase in enumerate(phases):
        # Get all scatterers belonging to one phase at a time, and the corresponding miller indices.
        grain_index = np.where(element_phase_map == i)[0]
        miller_indices = np.float32(phase.miller_indices)
        # Get all scattering vectors for all scatterers in a given phase
        G_0 = laue.get_G(orientation_lab[grain_index], eB[grain_index], miller_indices)
        # Now G_0 and rho_factors are sent before computation to save memory when diffracting many grains.
        reflection_index, time_values = (
            laue.find_solutions_to_tangens_half_angle_equation(
                G_0,
                rho_0_factor,
                rho_1_factor,
                rho_2_factor,
                rigid_body_motion.rotation_angle,
            )
        )
        G_0_reflected = G_0.transpose(0, 2, 1)[
            reflection_index[0, :], reflection_index[1, :]
        ]

        del G_0
        # We now assemble the dataframes with the valid reflections for each grain and phase including time, hkl plane and G vector

        table = pd.DataFrame(
            {
                "Grain": grain_index[reflection_index[0]],
                "phase": i,
                "hkl": reflection_index[1],
                "time": time_values,
                "G_0x": G_0_reflected[:, 0],
                "G_0y": G_0_reflected[:, 1],
                "G_0z": G_0_reflected[:, 2],
            }
        )

        del G_0_reflected, reflection_index, time_values
        reflections_df = pd.concat([reflections_df, table], axis=0).sort_values(
            by="Grain"
        )

    reflections_df = reflections_df[
        (0 < reflections_df["time"]) & (reflections_df["time"] < 1)
    ]  # We filter out the times which exceed 0 or 1
    reflections_df[["Gx", "Gy", "Gz"]] = rigid_body_motion.rotate(
        reflections_df[["G_0x", "G_0y", "G_0z"]].values, reflections_df["time"].values
    )
    reflections_df[["k'x", "k'y", "k'z"]] = (
        reflections_df[["Gx", "Gy", "Gz"]] + beam.wave_vector
    )
    reflections_df[["Source_x", "Source_y", "Source_z"]] = rigid_body_motion(
        espherecentroids[reflections_df["Grain"]], reflections_df["time"].values
    )
    reflections_df[["zd", "yd"]] = detector.get_intersection(
        reflections_df[["k'x", "k'y", "k'z"]].values,
        reflections_df[["Source_x", "Source_y", "Source_z"]].values,
    )
    reflections_df = reflections_df[
        detector.contains(reflections_df["zd"], reflections_df["yd"])
    ]

    element_vertices_0 = ecoord[reflections_df["Grain"]]
    element_vertices = rigid_body_motion(
        element_vertices_0, reflections_df["time"].values
    )

    reflections_np = (
        reflections_df.values
    )  # We move from pandas to numpy for enhanced speed
    scattering_units = []

    if BB_intersection:
        # A Bounding-Box intersection is a simplified way of computing the grains that interact with the beam (to enhance speed),
        # simply considering the beam as a prism and the tets that interact are those whose centroid is contained in the prism.

        reflections_np = reflections_np[
            reflections_np[:, 14] < (beam.vertices[:, 1].max())
        ]  #
        reflections_np = reflections_np[
            reflections_np[:, 14] > (beam.vertices[:, 1].min())
        ]  #
        reflections_np = reflections_np[
            reflections_np[:, 15] < (beam.vertices[:, 2].max())
        ]  #
        reflections_np = reflections_np[
            reflections_np[:, 15] > (beam.vertices[:, 2].min())
        ]  #

        for ei in range(reflections_np.shape[0]):
            scattering_unit = ScatteringUnit(
                ConvexHull(element_vertices[ei]),
                reflections_np[ei, 10:13],  # outgoing wavevector
                beam.wave_vector,
                beam.wavelength,
                beam.polarization_vector,
                rigid_body_motion.rotation_axis,
                reflections_np[ei, 3],  # time
                phases[reflections_np[ei, 1].astype(int)],  # phase
                reflections_np[ei, 2].astype(int),  # hkl index
                ei,
                zd=reflections_np[
                    ei, 16
                ],  # zd saved to avoid recomputing during redering
                yd=reflections_np[ei, 17],
            )  # yd saved to avoid recomputing during redering)

            scattering_units.append(scattering_unit)

    else:
        """Otherwise, compute the true intersection of each tet with the beam to get the true scattering volume."""
        for ei in range(element_vertices.shape[0]):
            scattering_region = beam.intersect(element_vertices[ei])

            if scattering_region is not None:
                scattering_unit = ScatteringUnit(
                    scattering_region,
                    reflections_np[ei, 10:13],  # outgoing wavevector
                    beam.wave_vector,
                    beam.wavelength,
                    beam.polarization_vector,
                    rigid_body_motion.rotation_axis,
                    reflections_np[ei, 3],  # time
                    phases[reflections_np[ei, 1].astype(int)],  # phase
                    reflections_np[ei, 2].astype(int),  # hkl index
                    ei,
                    zd=reflections_np[
                        ei, 16
                    ],  # zd saved to avoid recomputing during redering
                    yd=reflections_np[
                        ei, 17
                    ],  # yd saved to avoid recomputing during redering
                )

                scattering_units.append(scattering_unit)

    return scattering_units


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
        verbose=False,
        number_of_processes=1,
        number_of_frames=1,
        proximity=False,
        BB_intersection=False,
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
                to be collected. Defaults to 1, which means that the detector reads diffraction during the full rigid body
                motion and integrates out the signal to a single frame. The number_of_frames keyword primarily allows for single
                rotation axis full 180 dgrs or 360 dgrs sample rotation data sets to be computed rapidly and convinently.
            proximity (:obj:`bool`): Set to False if all or most grains from the sample are expected to diffract.
                For instance, if the diffraction scan illuminates all grains from the sample at least once at a give angle/position.
            BB_intersection (:obj:`bool`): Set to True in order to assume the beam as a square prism, the scattering volume for the tetrahedra
                to be the whole tetrahedron and the scattering tetrahedra to be all those whose centroids are included in the square prism.
                Greatly speeds up computation, valid approximation for powder-like samples.

        """
        if verbose and number_of_processes != 1:
            raise NotImplemented(
                "Verbose mode is not implemented for multiprocesses computations"
            )

        min_bragg_angle, max_bragg_angle = self._get_bragg_angle_bounds(
            detector, beam, min_bragg_angle, max_bragg_angle
        )

        for phase in self.phases:
            with utils._verbose_manager(verbose):
                phase.setup_diffracting_planes(
                    beam.wavelength, min_bragg_angle, max_bragg_angle
                )

        espherecentroids = np.array_split(
            self.mesh_lab.espherecentroids, number_of_processes, axis=0
        )
        eradius = np.array_split(self.mesh_lab.eradius, number_of_processes, axis=0)
        orientation_lab = np.array_split(
            self.orientation_lab, number_of_processes, axis=0
        )
        eB = np.array_split(self._eB, number_of_processes, axis=0)
        element_phase_map = np.array_split(
            self.element_phase_map, number_of_processes, axis=0
        )
        enod = np.array_split(self.mesh_lab.enod, number_of_processes, axis=0)

        args = []
        for i in range(number_of_processes):
            ecoord = np.zeros((enod[i].shape[0], 4, 3))
            for k, en in enumerate(enod[i]):
                ecoord[k, :, :] = self.mesh_lab.coord[en]
            args.append(
                {
                    "beam": beam,
                    "detector": detector,
                    "rigid_body_motion": rigid_body_motion,
                    "phases": self.phases,
                    "espherecentroids": espherecentroids[i],
                    "eradius": eradius[i],
                    "orientation_lab": orientation_lab[i],
                    "eB": eB[i],
                    "element_phase_map": element_phase_map[i],
                    "ecoord": ecoord,
                    "verbose": verbose,
                    "proximity": proximity,
                    "BB_intersection": BB_intersection,
                }
            )

        if number_of_processes == 1:
            all_scattering_units = _diffract(args[0])

        else:
            with Pool(number_of_processes) as p:
                scattering_units = p.map(_diffract, args)
            all_scattering_units = []
            for su in scattering_units:
                all_scattering_units.extend(su)

        if number_of_frames == 1:
            detector.frames.append(all_scattering_units)
        else:
            # TODO: unit test
            all_scattering_units.sort(
                key=lambda scattering_unit: scattering_unit.time, reverse=True
            )
            dt = 1.0 / number_of_frames
            start_time_of_current_frame = 0
            while start_time_of_current_frame <= 1 - 1e-8:
                frame = []
                while (
                    len(all_scattering_units) > 0
                    and all_scattering_units[-1].time < start_time_of_current_frame + dt
                ):
                    frame.append(all_scattering_units.pop())
                start_time_of_current_frame += dt
                detector.frames.append(frame)
            assert len(all_scattering_units) == 0

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

            misorientations = utils.get_misorientations(self.orientation_sample)

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
            return dill.load(f)

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
            mesh_nodes_contained_by_beam = self.mesh_lab.coord[
                beam.contains(self.mesh_lab.coord.T), :
            ]
            if mesh_nodes_contained_by_beam.shape[0] != 0:
                source_point = np.mean(mesh_nodes_contained_by_beam, axis=0)
            else:
                source_point = self.mesh_lab.centroid
            max_bragg_angle = detector.get_wrapping_cone(beam.wave_vector, source_point)
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
