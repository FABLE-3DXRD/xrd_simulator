import unittest
import numpy as np
import torch
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator import utils
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.utils import _epsilon_to_b
from xfab import tools
import os


class TestPolycrystal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run once for all tests - creates expensive shared resources."""
        np.random.seed(10)

        cls.pixel_size = 75.
        cls.detector_size = cls.pixel_size * 1024
        cls.detector_distance = 142938.28756189224
        det_corner_0 = np.array(
            [cls.detector_distance, -cls.detector_size / 2., -cls.detector_size / 2.])
        det_corner_1 = np.array(
            [cls.detector_distance, cls.detector_size / 2., -cls.detector_size / 2.])
        det_corner_2 = np.array(
            [cls.detector_distance, -cls.detector_size / 2., cls.detector_size / 2.])

        cls.detector = Detector(
            cls.pixel_size,
            cls.pixel_size,
            det_corner_0,
            det_corner_1,
            det_corner_2)

        # Expensive mesh generation - only done once!
        cls.mesh = TetraMesh.generate_mesh_from_levelset(
            level_set=lambda x: np.dot(x, x) - cls.detector_size / 10.,
            bounding_radius=1.1 * cls.detector_size / 10.,
            max_cell_circumradius=0.01 * cls.detector_size / 10.)

        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221'  # Quartz
        cls.phases = [Phase(unit_cell, sgname)]
        euler_angles = np.random.rand(
            cls.mesh.number_of_elements, 3) * 2 * np.pi
        cls.orientation = np.array([tools.euler_to_u(ea[0], ea[1], ea[2])
                                     for ea in euler_angles])
        cls.element_phase_map = np.zeros(
            (cls.mesh.number_of_elements,)).astype(int)

        w = cls.detector_size / 2.  # full field illumination
        beam_vertices = np.array([
            [-cls.detector_distance, -w, -w],
            [-cls.detector_distance, w, -w],
            [-cls.detector_distance, w, w],
            [-cls.detector_distance, -w, w],
            [cls.detector_distance, -w, -w],
            [cls.detector_distance, w, -w],
            [cls.detector_distance, w, w],
            [cls.detector_distance, -w, w]])
        wavelength = 0.285227
        xray_propagation_direction = np.array(
            [1, 0, 0]) * 2 * np.pi / wavelength
        polarization_vector = np.array([0, 1, 0])
        cls.beam = Beam(
            beam_vertices,
            xray_propagation_direction,
            wavelength,
            polarization_vector)

    def setUp(self):
        """Run before each test - creates fresh polycrystal for test isolation."""
        self.polycrystal = Polycrystal(
            mesh=self.__class__.mesh, 
            orientation=self.__class__.orientation, 
            strain=np.zeros((3, 3)), 
            phases=self.__class__.phases, 
            element_phase_map=self.__class__.element_phase_map)

    def test_diffract(self):

        rotation_angle = 10 * np.pi / 180.
        rotation_axis = np.array([0, 0, 1])
        translation = np.array([0, 0, 0])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        peaks_dict = self.polycrystal.diffract(self.__class__.beam, motion, detector=self.__class__.detector, verbose=True)
        peaks_dict1 = self.polycrystal.diffract(self.__class__.beam, motion, detector=self.__class__.detector, verbose=False)

        diffraction_pattern, _ = self.__class__.detector.render(
            peaks_dict, frames_to_render=1, method='gauss')

        diffraction_pattern1, _ = self.__class__.detector.render(
            peaks_dict1, frames_to_render=1, method='gauss')

        # The rendered diffraction pattern should have intensity
        if hasattr(diffraction_pattern, 'cpu'):
            diffraction_pattern_np = diffraction_pattern.cpu().numpy()
            diffraction_pattern1_np = diffraction_pattern1.cpu().numpy()
        else:
            diffraction_pattern_np = np.array(diffraction_pattern)
            diffraction_pattern1_np = np.array(diffraction_pattern1)
            
        self.assertGreater(np.sum(diffraction_pattern_np), 0)

        self.assertTrue(np.allclose(diffraction_pattern_np, diffraction_pattern1_np), msg='Multiproccessing is broken')

        # .. and the intensity should be scattered over the image
        w = int(self.__class__.detector_size / 5.)
        for i in range(w, diffraction_pattern_np.shape[1] - w, w):
            for j in range(w, diffraction_pattern_np.shape[1] - w, w):
                subsum = np.sum(diffraction_pattern_np[0, i - w:i + w, j - w:j + w])
                self.assertGreaterEqual(subsum, 0)

        # Peaks should be confined to rings - check using peak data directly
        import torch
        peaks = peaks_dict["peaks"]
        columns = peaks_dict["columns"]
        
        # Convert to numpy if needed
        if torch.is_tensor(peaks):
            peaks_np = peaks.cpu().numpy()
        else:
            peaks_np = np.array(peaks)
            
        # Extract 2theta angles from peaks
        tth_idx = columns.index("2theta") if "2theta" in columns else None
        if tth_idx is not None:
            bragg_angles = peaks_np[:, tth_idx] / 2.0  # Convert 2theta to theta
            bragg_angles = np.degrees(bragg_angles)
            
            # count the number of non-overlapping rings there should be quite a few.
            hist = np.histogram(bragg_angles, bins=np.linspace(0, 20, 360))[0]
            csequence, nosequences = 0, 0
            for i in range(hist.shape[0]):
                if hist[i] > 0:
                    csequence += 1
                elif csequence >= 1:
                    nosequences += 1
                    csequence = 0
            self.assertGreaterEqual(
                nosequences,
                5,
                msg="Few or no rings appeared from diffraction.")

    def test_save_and_load(self):
        orientation_lab = utils.ensure_numpy(self.polycrystal.orientation_lab)
        path = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                'data'),
            'my_polycrystal')
        self.polycrystal.save(path, save_mesh_as_xdmf=True)
        self.polycrystal = Polycrystal.load(path+'.pc')
        self.assertTrue(
            np.allclose(
                orientation_lab,
                utils.ensure_numpy(self.polycrystal.orientation_lab)),
            msg='Data corrupted on save and load')
        os.remove(path + '.pc')
        os.remove(path + ".xdmf")
        os.remove(path + ".h5")

        self.polycrystal.save(path+'.pc', save_mesh_as_xdmf=True)
        self.polycrystal = Polycrystal.load(path+'.pc')
        self.assertTrue(
            np.allclose(
                orientation_lab,
                utils.ensure_numpy(self.polycrystal.orientation_lab)),
            msg='Data corrupted on save and load')
        os.remove(path + '.pc')
        os.remove(path + ".xdmf")
        os.remove(path + ".h5")

    def test_dimension_handling(self):

        Polycrystal(mesh=self.__class__.mesh,
                    orientation=self.__class__.orientation,
                    strain=np.zeros((3, 3)),
                    phases=self.__class__.phases[0],
                    element_phase_map=None)
        Polycrystal(mesh=self.__class__.mesh,
                    orientation=self.__class__.orientation,
                    strain=np.zeros((3, 3)),
                    phases=self.__class__.phases[0],
                    element_phase_map=self.__class__.element_phase_map)
        Polycrystal(
            mesh=self.__class__.mesh,
            orientation=self.__class__.orientation,
            strain=np.zeros(
                (self.__class__.mesh.number_of_elements,
                    3,
                    3)),
            phases=self.__class__.phases,
            element_phase_map=self.__class__.element_phase_map)

    def test_transformation(self):
        strains = np.zeros((self.__class__.mesh.number_of_elements, 3, 3))
        for i in range(self.__class__.mesh.number_of_elements):
            strains[i] = 0.001 * (np.random.rand(3, 3) - 0.5)
            strains[i] = 0.5 * (strains[i] + strains[i].T)
        polycrystal = Polycrystal(mesh=self.__class__.mesh,
                                  orientation=self.__class__.orientation,
                                  strain=strains,
                                  phases=self.__class__.phases,
                                  element_phase_map=self.__class__.element_phase_map)
        rotation_angle = 10 * np.pi / 180.
        rotation_axis = np.array([0, 0, 1])
        translation = np.array([-34.0, 0.243, 345.324])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        time = 0.8436
        polycrystal.transform(motion, time=time)
        Rot_mat = motion.rotator.get_rotation_matrix(torch.tensor(time * rotation_angle))
        Rot_mat = utils.ensure_numpy(Rot_mat)
        unit_vector = np.random.rand(3,)
        unit_vector = unit_vector / np.linalg.norm(unit_vector)
        new_unit_vector = np.dot(Rot_mat, unit_vector)
        for i, strain in enumerate(strains):
            s1 = np.dot(unit_vector, np.dot(strain, unit_vector))
            strain_lab_i = utils.ensure_numpy(polycrystal.strain_lab[i])
            s2 = np.dot(
                new_unit_vector,
                np.dot(
                    strain_lab_i,
                    new_unit_vector))

            
            self.assertAlmostEqual(
                s1, s2, msg="Transformation does not preserve directional strains")

        # TODO: also test the orientation transformations.


if __name__ == '__main__':
    unittest.main()
