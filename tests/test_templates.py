import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from xrd_simulator import templates, utils
import matplotlib.pyplot as plt

class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(5)  # changes all randomization in the test

    def test_s3dxrd(self):

        parameters = {
            "detector_distance": 191023.9164,
            "detector_center_pixel_z": 256.2345,
            "detector_center_pixel_y": 255.1129,
            "pixel_side_length_z": 181.4234,
            "pixel_side_length_y": 180.2343,
            "number_of_detector_pixels_z": 512,
            "number_of_detector_pixels_y": 512,
            "wavelength": 0.285227,
            "beam_side_length_z": 512 * 200.,
            "beam_side_length_y": 512 * 200.,
            "rotation_step": np.radians(1.634),
            "rotation_axis": np.array([0., 0., 1.0])
        }

        beam, detector, motion = templates.s3dxrd(parameters)

        beam_centroid = utils.ensure_numpy(beam.centroid)
        for ci in beam_centroid:
            self.assertAlmostEqual(ci, 0, msg="beam not at origin.")

        det_approx_centroid = utils.ensure_numpy(detector.det_corner_0).copy()
        det_approx_centroid[1] += utils.ensure_numpy(detector.det_corner_1)[1]
        det_approx_centroid[2] += utils.ensure_numpy(detector.det_corner_2)[2]

        self.assertAlmostEqual(
            det_approx_centroid[0],
            parameters["detector_distance"],
            msg="Detector distance wrong.")
        self.assertLessEqual(
            np.abs(
                det_approx_centroid[1]),
            5 * parameters["pixel_side_length_y"],
            msg="Detector not centered.")
        self.assertLessEqual(
            np.abs(
                det_approx_centroid[2]),
            5 * parameters["pixel_side_length_z"],
            msg="Detector not centered.")

        original_vector = np.random.rand(3,) - 0.5
        time = 0.234986
        transformed_vector = utils.ensure_numpy(motion(original_vector, time))

        angle = parameters["rotation_step"] * time
        s, c = np.sin(angle), np.cos(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        self.assertAlmostEqual(transformed_vector[0], np.dot(Rz, original_vector)[
                               0], msg="Motion does not rotate around z-axis")
        self.assertAlmostEqual(transformed_vector[1], np.dot(Rz, original_vector)[
                               1], msg="Motion does not rotate around z-axis")
        self.assertAlmostEqual(
            transformed_vector[2],
            original_vector[2],
            msg="Motion does not rotate around z-axis")

    def test_polycrystal_from_odf(self):

        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221'  # Quartz

        def orientation_density_function(
            x, q): return 1. / (np.pi**2)  # uniform ODF
        number_of_crystals = 500
        sample_bounding_cylinder_height = 50
        sample_bounding_cylinder_radius = 25
        maximum_sampling_bin_seperation = np.radians(10.0)
        # Linear strain gradient along rotation axis.
        def strain_tensor(x): return np.array(
            [[0, 0, 0.02 * x[2] / sample_bounding_cylinder_height], [0, 0, 0], [0, 0, 0]])

        polycrystal = templates.polycrystal_from_odf(
            orientation_density_function,
            number_of_crystals,
            sample_bounding_cylinder_height,
            sample_bounding_cylinder_radius,
            unit_cell,
            sgname,
            path_to_cif_file=None,
            maximum_sampling_bin_seperation=maximum_sampling_bin_seperation,
            strain_tensor=strain_tensor)

        # Compare Euler angle distributions to scipy random uniform orientation
        # sampler
        orientation_lab_np = utils.ensure_numpy(polycrystal.orientation_lab)
        euler1 = np.array([Rotation.from_matrix(U).as_euler(
            'xyz', degrees=True) for U in orientation_lab_np])
        euler2 = Rotation.random(10 * euler1.shape[0]).as_euler('xyz')

        for i in range(3):
            hist1, bins = np.histogram(euler1[:, i])
            hist2, bins = np.histogram(euler2[:, i])
            hist2 = hist2 / 10.
            # These histograms should look roughly the same
            self.assertLessEqual(
                np.max(
                    np.abs(
                        hist1 -
                        hist2)),
                number_of_crystals *
                0.05,
                "ODF not sampled correctly.")

        parameters = {
            "detector_distance": 191023.9164,
            "detector_center_pixel_z": 256.2345,
            "detector_center_pixel_y": 255.1129,
            "pixel_side_length_z": 181.4234,
            "pixel_side_length_y": 180.2343,
            "number_of_detector_pixels_z": 512,
            "number_of_detector_pixels_y": 512,
            "wavelength": 0.285227,
            "beam_side_length_z": 512 * 200.,
            "beam_side_length_y": 512 * 200.,
            "rotation_step": np.radians(20.0),
            "rotation_axis": np.array([0., 0., 1.0])
        }

        beam, detector, motion = templates.s3dxrd(parameters)

        number_of_crystals = 100
        sample_bounding_cylinder_height = 256 * 180 / 128.
        sample_bounding_cylinder_radius = 256 * 180 / 128.

        polycrystal = templates.polycrystal_from_odf(
            orientation_density_function,
            number_of_crystals,
            sample_bounding_cylinder_height,
            sample_bounding_cylinder_radius,
            unit_cell,
            sgname,
            path_to_cif_file=None,
            maximum_sampling_bin_seperation=maximum_sampling_bin_seperation,
            strain_tensor=strain_tensor)

        polycrystal.transform(motion, time=0.134)
        peaks_dict = polycrystal.diffract(
            beam,
            motion,
            min_bragg_angle=0,
            max_bragg_angle=None,
            detector=detector,
            verbose=True)

        diffraction_pattern, _ = detector.render(
            peaks_dict,
            frames_to_render=1,
            method="micro")
        
        # Convert to numpy for analysis
        if hasattr(diffraction_pattern, 'cpu'):
            diffraction_pattern_np = diffraction_pattern[0].cpu().numpy()
        else:
            diffraction_pattern_np = np.array(diffraction_pattern[0])
        
        # Check for ring patterns by analyzing radial intensity distribution
        # instead of using deprecated _diffractogram
        det_center_z = parameters['detector_center_pixel_z']
        det_center_y = parameters['detector_center_pixel_y']
        
        # Create radial profile
        m, n = diffraction_pattern_np.shape
        max_radius = int(min(m, n) / 2)
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius)
        
        for i in range(m):
            for j in range(n):
                radius = int(np.sqrt((i - det_center_z)**2 + (j - det_center_y)**2))
                if radius < max_radius:
                    radial_profile[radius] += diffraction_pattern_np[i, j]
                    radial_counts[radius] += 1
        
        radial_profile = radial_profile / np.maximum(radial_counts, 1)
        radial_profile[radial_profile < 0.5 * np.median(radial_profile)] = 0
        
        csequence, nosequences = 0, 0
        for i in range(len(radial_profile)):
            if radial_profile[i] > 0:
                csequence += 1
            elif csequence >= 1:
                nosequences += 1
                csequence = 0
        self.assertGreaterEqual(
            nosequences,
            5,
            msg="Few or no rings appeared from diffraction.")

    def test_get_uniform_powder_sample(self):
        sample_bounding_radius = 256 * 180 / 128.
        polycrystal = templates.get_uniform_powder_sample(
            sample_bounding_radius,
            number_of_grains=50,
            unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
            sgname='P3221',
            strain_tensor=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.01]])
        )
        for c in polycrystal.mesh_lab.coord:
            self.assertLessEqual(
                torch.linalg.norm(c).item(),
                sample_bounding_radius + 1e-8,
                msg='Powder sample not contained by bounding sphere.')

        parameters = {
            "detector_distance": 191023.9164,
            "detector_center_pixel_z": 256.2345,
            "detector_center_pixel_y": 255.1129,
            "pixel_side_length_z": 181.4234,
            "pixel_side_length_y": 180.2343,
            "number_of_detector_pixels_z": 512,
            "number_of_detector_pixels_y": 512,
            "wavelength": 0.285227,
            "beam_side_length_z": 512 * 200.,
            "beam_side_length_y": 512 * 200.,
            "rotation_step": np.radians(20.0),
            "rotation_axis": np.array([0., 0., 1.0])
        }

        beam, detector, motion = templates.s3dxrd(parameters)
        polycrystal.transform(motion, time=0.234)

        peaks_dict = polycrystal.diffract(
            beam,
            motion,
            min_bragg_angle=0,
            max_bragg_angle=None,
            detector=detector,
            verbose=True)

        diffraction_pattern, _ = detector.render(
            peaks_dict,
            frames_to_render=1,
            method="micro")

        # Convert to numpy for analysis
        if hasattr(diffraction_pattern, 'cpu'):
            diffraction_pattern_np = diffraction_pattern[0].cpu().numpy()
        else:
            diffraction_pattern_np = np.array(diffraction_pattern[0])
        
        # Create radial profile for ring detection
        det_center_z = parameters['detector_center_pixel_z']
        det_center_y = parameters['detector_center_pixel_y']
        
        m, n = diffraction_pattern_np.shape
        max_radius = int(min(m, n) / 2)
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius)
        
        for i in range(m):
            for j in range(n):
                radius = int(np.sqrt((i - det_center_z)**2 + (j - det_center_y)**2))
                if radius < max_radius:
                    radial_profile[radius] += diffraction_pattern_np[i, j]
                    radial_counts[radius] += 1
        
        radial_profile = radial_profile / np.maximum(radial_counts, 1)
        radial_profile[radial_profile < 0.5 * np.median(radial_profile)] = 0

        csequence, nosequences = 0, 0
        for i in range(len(radial_profile)):
            if radial_profile[i] > 0:
                csequence += 1
            elif csequence >= 1:
                nosequences += 1
                csequence = 0

        self.assertGreaterEqual(
            nosequences,
            5,
            msg="Few or no rings appeared from diffraction.")


if __name__ == '__main__':
    unittest.main()
