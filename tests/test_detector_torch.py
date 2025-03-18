import unittest
from scipy.spatial import ConvexHull
import os
import torch
from xrd_simulator.detector import Detector
from xrd_simulator.phase import Phase
from xrd_simulator.scattering_unit import ScatteringUnit
from xrd_simulator.utils import ensure_torch, ensure_numpy
import numpy as np

torch.set_default_dtype(torch.float64)


class TestDetector(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(10)
        self.pixel_size_z = ensure_torch(50.0)
        self.pixel_size_y = ensure_torch(40.0)
        self.detector_size = ensure_torch(10000.0)
        self.det_corner_0 = ensure_torch([1, 0, 0]) * self.detector_size
        self.det_corner_1 = ensure_torch([1, 1, 0]) * self.detector_size
        self.det_corner_2 = ensure_torch([1, 0, 1]) * self.detector_size
        self.detector = Detector(
            self.pixel_size_z,
            self.pixel_size_y,
            self.det_corner_0,
            self.det_corner_1,
            self.det_corner_2,
        )

    def test_init(self):

        for o, otrue in zip(
            self.detector.det_corner_0, torch.tensor([1, 0, 0]) * self.detector_size
        ):
            self.assertAlmostEqual(o, otrue, msg="detector origin is incorrect")

        for z, ztrue in zip(self.detector.zdhat, torch.tensor([0, 0, 1])):
            self.assertAlmostEqual(z, ztrue, msg="zdhat is incorrect")

        for y, ytrue in zip(self.detector.ydhat, torch.tensor([0, 1, 0])):
            self.assertAlmostEqual(y, ytrue, msg="ydhat is incorrect")

        self.assertAlmostEqual(
            self.detector.zmax,
            self.detector_size,
            msg="Bad detector dimensions in zmax",
        )
        self.assertAlmostEqual(
            self.detector.ymax,
            self.detector_size,
            msg="Bad detector dimensions in ymax",
        )

        for n, ntrue in zip(self.detector.normal, torch.tensor([-1, 0, 0])):
            self.assertAlmostEqual(n, ntrue, msg="Bad detector normal")

    def test_centroid_render(self):
        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = (
            ensure_torch([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + v * torch.sqrt(torch.tensor(2.0)) * self.detector_size / 2.0
        )  # tetra at detector centre
        ch1 = ConvexHull(ensure_numpy(verts1))
        verts2 = (
            ensure_torch([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + 2 * v * torch.sqrt(torch.tensor(2.0)) * self.detector_size
        )  # tetra out of detector bounds
        ch2 = ConvexHull(ensure_numpy(verts2))
        wavelength = 1.0

        incident_wave_vector = 2 * torch.pi * ensure_torch([1, 0, 0]) / wavelength
        scattered_wave_vector = (
            self.det_corner_0
            + self.pixel_size_y * 3 * self.detector.ydhat
            + self.pixel_size_z * 2 * self.detector.zdhat
        )
        scattered_wave_vector = (
            2
            * torch.pi
            * scattered_wave_vector
            / (torch.linalg.norm(scattered_wave_vector) * wavelength)
        )

        scattered_wave_vector = scattered_wave_vector.unsqueeze(0)  # Add batch dim
        zd1, yd1, _ = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0, keepdim=True)
            )[0]
        )
        zd2, yd2, _ = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts2.mean(dim=0)[None, :]
            )[0]
        )

        data = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "Fe_mp-150_conventional_standard.cif",
        )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = "Fm-3m"  # Iron
        phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase.setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)

        scattering_unit1 = ScatteringUnit(
            ch1,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=ensure_torch([0, 1, 0]),
            rotation_axis=ensure_torch([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd1,
            yd=yd1,
        )

        scattering_unit2 = ScatteringUnit(
            ch2,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=ensure_torch([0, 1, 0]),
            rotation_axis=ensure_torch([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd2,
            yd=yd2,
        )

        self.detector.frames.append([scattering_unit1, scattering_unit2])
        diffraction_pattern = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="centroid",
        )

        # the sample sits at the centre of the detector.
        expected_z_pixel = int(self.detector_size / (2 * self.pixel_size_z)) + 2
        expected_y_pixel = int(self.detector_size / (2 * self.pixel_size_y)) + 3

        dy = self.detector._point_spread_kernel_shape[0]
        dz = self.detector._point_spread_kernel_shape[1]
        active_det_part = diffraction_pattern[
            expected_z_pixel - dy + 1 : expected_z_pixel + dy,
            expected_y_pixel - dz + 1 : expected_y_pixel + dz,
        ]

        self.assertAlmostEqual(
            torch.sum(active_det_part),
            ch1.volume,
            msg="detector rendering did not capture scattering_unit",
        )
        self.assertAlmostEqual(
            torch.sum(diffraction_pattern),
            ch1.volume,
            msg="detector rendering captured out of bounds scattering_unit",
        )

        # Try rendering with advanced intensity model
        diffraction_pattern = self.detector.render(
            frames_to_render=0, lorentz=True, polarization=False, structure_factor=False
        )

        self.assertTrue(
            diffraction_pattern[expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not use lorentz factor",
        )

        diffraction_pattern = self.detector.render(
            frames_to_render=0, lorentz=False, polarization=True, structure_factor=False
        )
        self.assertTrue(
            diffraction_pattern[expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not use polarization factor",
        )

        diffraction_pattern = self.detector.render(
            frames_to_render=0, lorentz=False, polarization=False, structure_factor=True
        )
        self.assertTrue(
            diffraction_pattern[expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not use structure_factor factor",
        )

    def test_centroid_render_with_scintillator(self):
        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = (
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + v * torch.sqrt(torch.tensor(2.0)) * self.detector_size / 2.0
        )  # tetra at detector center
        ch1 = ConvexHull(ensure_numpy(verts1))
        verts2 = (
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + 2 * v * torch.sqrt(torch.tensor(2.0)) * self.detector_size
        )  # tetra out of detector bounds
        ch2 = ConvexHull(ensure_numpy(verts2))
        wavelength = 1.0

        incident_wave_vector = 2 * torch.pi * torch.tensor([1, 0, 0]) / (wavelength)
        scattered_wave_vector = (
            self.det_corner_0
            + self.pixel_size_y * 3 * self.detector.ydhat
            + self.pixel_size_z * 2 * self.detector.zdhat
        )
        scattered_wave_vector = (
            2
            * torch.pi
            * scattered_wave_vector
            / (torch.linalg.norm(scattered_wave_vector) * wavelength)
        )

        zd1, yd1 = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0)[None, :]
            )[0]
        )
        zd2, yd2 = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts2.mean(dim=0)[None, :]
            )[0]
        )

        data = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "Fe_mp-150_conventional_standard.cif",
        )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = "Fm-3m"  # Iron
        phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase.setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)

        scattering_unit1 = ScatteringUnit(
            ch1,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd1,
            yd=yd1,
        )
        scattering_unit2 = ScatteringUnit(
            ch2,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd2,
            yd=yd2,
        )
        self.detector.frames.append([scattering_unit1, scattering_unit2])
        self.detector.point_spread_kernel_shape = (3, 3)
        diffraction_pattern = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="centroid_with_scintillator",
        )

        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = (
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            + v * torch.sqrt(torch.tensor(2.0)) * self.detector_size / 2.0
            + self.pixel_size_z * 0.1
        )  # tetra at detector center perturbed
        ch1 = ConvexHull(ensure_numpy(verts1))
        zd1, yd1 = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0)[None, :]
            )[0]
        )

        scattering_unit1 = ScatteringUnit(
            ch1,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd1,
            yd=yd1,
        )
        self.detector.frames[-1] = [scattering_unit1, scattering_unit2]
        diffraction_pattern_2 = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="centroid_with_scintillator",
        )

        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = (
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
            - torch.tensor([0, 1.5, 2.0]) * self.pixel_size_z
        )
        ch1 = ConvexHull(ensure_numpy(verts1))
        zd1, yd1 = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0)[None, :]
            )[0]
        )

        scattering_unit1 = ScatteringUnit(
            ch1,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd1,
            yd=yd1,
        )
        self.detector.frames[-1] = [scattering_unit1, scattering_unit2]
        diffraction_pattern_3 = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="centroid_with_scintillator",
        )

        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
        ) + torch.tensor([0, 246 * self.pixel_size_y, 196 * self.pixel_size_z])
        ch1 = ConvexHull(ensure_numpy(verts1))
        zd1, yd1 = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0)[None, :]
            )[0]
        )

        scattering_unit1 = ScatteringUnit(
            ch1,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd1,
            yd=yd2,
        )
        self.detector.frames[-1] = [scattering_unit1, scattering_unit2]
        diffraction_pattern_3 = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="centroid_with_scintillator",
        )

        pixels = diffraction_pattern[diffraction_pattern != 0]
        self.assertEqual(
            len(pixels.flatten()),
            (self.detector.point_spread_kernel_shape[0] + 2)
            * (self.detector.point_spread_kernel_shape[1] + 2),
        )

        # the sample sits at the centre of the detector.
        expected_z_pixel = int(self.detector_size / (2 * self.pixel_size_z)) + 2
        expected_y_pixel = int(self.detector_size / (2 * self.pixel_size_y)) + 3
        self.assertNotEqual(
            diffraction_pattern[expected_z_pixel, expected_y_pixel],
            diffraction_pattern_2[expected_z_pixel, expected_y_pixel],
        )

        self.assertEqual(
            torch.max(diffraction_pattern),
            diffraction_pattern[expected_z_pixel, expected_y_pixel],
        )

        self.assertEqual(
            torch.max(diffraction_pattern),
            diffraction_pattern[expected_z_pixel, expected_y_pixel],
        )

        self.assertEqual(
            torch.max(diffraction_pattern),
            diffraction_pattern[expected_z_pixel, expected_y_pixel],
        )

        dy = self.detector._point_spread_kernel_shape[0]
        dz = self.detector._point_spread_kernel_shape[1]
        active_det_part = diffraction_pattern[
            expected_z_pixel - dy + 1 : expected_z_pixel + dy,
            expected_y_pixel - dz + 1 : expected_y_pixel + dz,
        ]

        self.assertAlmostEqual(
            torch.sum(active_det_part),
            ch1.volume,
            msg="detector rendering did not capture scattering_unit",
        )
        self.assertAlmostEqual(
            torch.sum(diffraction_pattern),
            ch1.volume,
            msg="detector rendering captured out of bounds scattering_unit",
        )

        # Try rendering with advanced intensity model
        diffraction_pattern = self.detector.render(
            frames_to_render=0,
            lorentz=True,
            polarization=False,
            structure_factor=False,
            method="centroid_with_scintillator",
        )

        self.assertTrue(
            diffraction_pattern[expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not use lorentz factor",
        )

        diffraction_pattern = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=True,
            structure_factor=False,
            method="centroid_with_scintillator",
        )
        self.assertTrue(
            diffraction_pattern[expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not use polarization factor",
        )

        diffraction_pattern = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=True,
            method="centroid_with_scintillator",
        )
        self.assertTrue(
            diffraction_pattern[expected_z_pixel, expected_y_pixel] != ch1.volume,
            msg="detector rendering did not use structure_factor factor",
        )
        self.detector.point_spread_kernel_shape = (5, 5)

    def test_projection_render(self):

        # Convex hull of a sphere placed at the centre of the detector
        phi, theta = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, 25),
            torch.linspace(0, 2 * torch.pi, 25),
            indexing="ij",
        )
        r = 1.0 * self.detector_size / 4.0
        x = r * torch.cos(phi) * torch.sin(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(theta)
        hull_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=0).T
        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        sphere_hull = ConvexHull(
            ensure_numpy(hull_points)
            + ensure_numpy(v) * np.sqrt(2.0) * ensure_numpy(self.detector_size) / 2.0
        )

        # The spherical scattering_unit forward scatters.
        wavelength = 1.0
        incident_wave_vector = 2 * torch.pi * torch.tensor([1, 0, 0]) / wavelength
        scattered_wave_vector = 2 * torch.pi * torch.tensor([1, 0, 0]) / wavelength
        scattered_wave_vector = (
            2
            * torch.pi
            * scattered_wave_vector
            / (torch.linalg.norm(scattered_wave_vector) * wavelength)
        )

        # The spherical scattering_unit is composed of Fe_mp-150 (a pure iron
        # crystal)
        data = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "Fe_mp-150_conventional_standard.cif",
        )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = "Fm-3m"  # Iron
        phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase.setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)

        zd1, yd1 = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, hull_points.mean(dim=0)[None, :]
            )[0]
        )

        scattering_unit = ScatteringUnit(
            sphere_hull,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd1,
            yd=yd1,
        )

        self.detector.frames.append([scattering_unit])
        diffraction_pattern = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="project",
        )

        diffraction_pattern_parallel = self.detector.render(
            frames_to_render=0,
            lorentz=False,
            polarization=False,
            structure_factor=False,
            method="project",
            number_of_processes=2,
            verbose=False,
        )

        self.assertTrue(
            torch.allclose(diffraction_pattern, diffraction_pattern_parallel),
            msg="parallel rendering is broken",
        )

        projected_summed_intensity = torch.sum(diffraction_pattern)
        relative_error = (
            torch.abs(scattering_unit.volume - projected_summed_intensity)
            / scattering_unit.volume
        )
        self.assertLessEqual(
            relative_error, 1e-4, msg="Projected mass does not match the hull volume"
        )

        index = torch.where(diffraction_pattern == torch.max(diffraction_pattern))
        self.assertEqual(
            index[0][0],
            self.detector_size // (2 * self.pixel_size_z),
            msg="Projected mass does not match the hull volume",
        )
        self.assertEqual(
            index[1][0],
            self.detector_size // (2 * self.pixel_size_y),
            msg="Projected mass does not match the hull volume",
        )

        no_pixels_z = int(self.detector_size // self.pixel_size_z)
        no_pixels_y = int(self.detector_size // self.pixel_size_y)

        sz, sy = self.detector.point_spread_kernel_shape
        point_spread_padding = torch.max(
            torch.tensor([self.pixel_size_z * sz, self.pixel_size_y * sy])
        )
        for i in range(no_pixels_z):
            for j in range(no_pixels_y):
                zd = i * self.pixel_size_z
                yd = j * self.pixel_size_y
                if (zd - self.detector_size / 2.0) ** 2 + (
                    yd - self.detector_size / 2.0
                ) ** 2 > (1.01 * r + point_spread_padding) ** 2:
                    self.assertAlmostEqual(diffraction_pattern[i, j], 0)
                elif (zd - self.detector_size / 2.0) ** 2 + (
                    yd - self.detector_size / 2.0
                ) ** 2 < (0.99 * r) ** 2:
                    self.assertGreater(diffraction_pattern[i, j], 0)

    def test_get_intersection(self):

        # central normal algined ray
        ray_direction = torch.tensor([[2.23, 0.0, 0.0]])  # Add batch dim
        source_point = torch.tensor([[0.0, 0.0, 0.0]])  # Add batch dim

        z, y = tuple(
            self.detector.get_intersection(ray_direction, source_point[None, :])[0]
        )
        self.assertAlmostEqual(
            z, 0, msg="central detector-normal algined ray does not intersect at 0"
        )
        self.assertAlmostEqual(
            y, 0, msg="central detector-normal algined ray does not intersect at 0"
        )

        # translate the ray
        source_point += self.detector.ydhat * self.pixel_size_y
        source_point -= self.detector.zdhat * 2 * self.pixel_size_z
        z, y = tuple(
            self.detector.get_intersection(ray_direction, source_point[None, :])[0]
        )
        self.assertAlmostEqual(
            z,
            -2 * self.pixel_size_z,
            msg="translated detector-normal algined ray does not intersect properly",
        )
        self.assertAlmostEqual(
            y,
            self.pixel_size_y,
            msg="translated detector-normal algined ray does not intersect properly",
        )

        # tilt the ray
        ang = torch.atan(self.pixel_size_y / self.detector_size)
        frac = torch.tan(ang) * torch.linalg.norm(ray_direction)
        ray_direction += self.detector.ydhat * frac * 3
        z, y = tuple(
            self.detector.get_intersection(ray_direction, source_point[None, :])[0]
        )
        self.assertAlmostEqual(
            z,
            -2 * self.pixel_size_z,
            msg="translated and tilted ray does not intersect properly",
        )
        self.assertAlmostEqual(
            y,
            4 * self.pixel_size_y,
            msg="translated and tilted ray does not intersect properly",
        )

    def test_contains(self):
        c1 = self.detector.contains(self.detector_size / 10.0, self.detector_size / 5.0)
        self.assertTrue(c1, msg="detector does no contain included point")
        c2 = self.detector.contains(-self.detector_size / 8.0, self.detector_size / 3.0)
        self.assertTrue(not c2, msg="detector contain negative points")
        c4 = self.detector.contains(
            self.detector_size * 2 * self.pixel_size_z, self.detector_size / 374.0
        )
        self.assertTrue(not c4, msg="detector contain out of bounds points")

    def test_get_wrapping_cone(self):
        wavelength = 1.0
        k = 2 * torch.pi * torch.tensor([1, 0, 0]) / wavelength
        source_point = (
            (self.detector.zdhat + self.detector.ydhat) * self.detector_size / 2.0
        )
        opening_angle = self.detector.get_wrapping_cone(k, source_point)

        normed_det_center = (source_point + self.det_corner_0) / torch.linalg.norm(
            source_point + self.det_corner_0
        )
        normed_det_origin = self.det_corner_0 / torch.linalg.norm(self.det_corner_0)
        expected_angle = (
            torch.acos(torch.dot(normed_det_center, normed_det_origin)) / 2.0
        )

        self.assertAlmostEqual(
            opening_angle,
            expected_angle,
            msg="detector centered wrapping cone has faulty opening angle",
        )

        source_point = (
            (self.detector.zdhat + self.detector.ydhat) * self.detector_size / 2.0
        )
        source_point -= self.detector.zdhat * 10 * self.pixel_size_z
        source_point -= self.detector.ydhat * 10 * self.pixel_size_y
        opening_angle = self.detector.get_wrapping_cone(k, source_point)
        self.assertGreaterEqual(
            opening_angle,
            expected_angle,
            msg="detector off centered wrapping cone has opening angle",
        )

    def test_point_spread_kernel_shape(self):
        bad_kernel_shape = (6, 3)
        try:
            self.detector.point_spread_kernel_shape = bad_kernel_shape
        except ValueError:
            pass
        except:
            raise ValueError(
                "detector.point_spread_kernel_shape wrongly accepts even shapes."
            )

        good_kernel_shape = (5, 7)
        try:
            self.detector.point_spread_kernel_shape = good_kernel_shape
        except ValueError:
            raise ValueError(
                "detector.point_spread_kernel_shape wrongly rejects odd shapes."
            )
        except:
            pass

    def test_point_spread_function(self):
        default_kernel = self.detector._get_point_spread_function_kernel()
        default_kernel = torch.as_tensor(default_kernel)  # Convert to tensor
        self.assertAlmostEqual(
            default_kernel[2, 2].item(),
            torch.max(default_kernel).item(),
            msg="Default kernel appears to not be Gaussian",
        )
        self.assertAlmostEqual(
            default_kernel[0, 0].item(),
            torch.min(default_kernel).item(),
            msg="Default kernel appears to not be Gaussian",
        )
        self.assertAlmostEqual(
            default_kernel[2, 3].item(),
            default_kernel[2, 1].item(),
            msg="Default kernel appears to not be Gaussian",
        )
        self.assertAlmostEqual(
            default_kernel[3, 2].item(),
            default_kernel[1, 2].item(),
            msg="Default kernel appears to not be Gaussian",
        )
        self.assertAlmostEqual(
            torch.sum(default_kernel).item(),
            1,
            msg="Default kernel appears to not be Gaussian",
        )

        self.detector.point_spread_function = lambda z, y: torch.abs(y)
        self.detector.point_spread_kernel_shape = (5, 7)
        kernel = self.detector._get_point_spread_function_kernel()

        self.assertAlmostEqual(
            torch.sum(kernel),
            1,
            msg="Point spread function must be intensity preserving.",
        )
        for i in range(5):
            self.assertAlmostEqual(
                kernel[i, 3],
                0,
                msg="Point spread function must be intensity preserving.",
            )
            self.assertAlmostEqual(
                kernel[i, 4],
                1.0 / 60.0,
                msg="Point spread function kernel does not match spread function",
            )
            self.assertAlmostEqual(
                kernel[i, 5],
                2.0 / 60.0,
                msg="Point spread fframeunction kernel does not match spread function",
            )
            self.assertAlmostEqual(
                kernel[i, 6],
                3.0 / 60.0,
                msg="Point spread function kernel does not match spread function",
            )

    def test_save_and_load(self):
        self.detector.point_spread_function = lambda z, y: 1.0
        path = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"), "my_detector"
        )
        self.detector.save(path)
        self.detector = Detector.load(path + ".det")
        self.assertAlmostEqual(
            self.detector.point_spread_function(-23.0, 2.0),
            1.0,
            msg="Data corrupted on save and load",
        )
        os.remove(path + ".det")

    def test_eta0_render(self):
        v = self.detector.ydhat + self.detector.zdhat
        v = v / torch.linalg.norm(v)
        verts1 = (
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]) * 0.0000001
            + v * torch.sqrt(torch.tensor(2.0)) * self.detector_size / 2.0
            + 0.001 * self.detector.ydhat
            + 0.001 * self.detector.zdhat
        )  # tetra at detector centre
        ch = ConvexHull(ensure_numpy(verts1))

        wavelength = 1.0

        incident_wave_vector = 2 * torch.pi * torch.tensor([1, 0, 0]) / (wavelength)
        scattered_wave_vector = (
            2
            * torch.pi
            * torch.tensor([1, 0, 0.1])
            / (torch.sqrt(torch.tensor(0.1 * 0.1 + 1.0)) * wavelength)
        )

        data = os.path.join(
            os.path.join(os.path.dirname(__file__), "data"),
            "Fe_mp-150_conventional_standard.cif",
        )
        unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
        sgname = "Fm-3m"  # Iron
        phase = Phase(unit_cell, sgname, path_to_cif_file=data)
        phase.setup_diffracting_planes(wavelength, 0, 20 * torch.pi / 180)
        zd, yd = tuple(
            self.detector.get_intersection(
                scattered_wave_vector, verts1.mean(dim=0)[None, :]
            )[0]
        )
        scattering_unit = ScatteringUnit(
            ch,
            scattered_wave_vector=scattered_wave_vector,
            incident_wave_vector=incident_wave_vector,
            wavelength=wavelength,
            incident_polarization_vector=torch.tensor([0, 1, 0]),
            rotation_axis=torch.tensor([0, 0, 1]),
            time=0,
            phase=phase,
            hkl_indx=0,
            element_index=0,
            zd=zd,
            yd=yd,
        )

        self.detector.frames.append([scattering_unit])

        for method in ["project", "centroid"]:
            diffraction_pattern = self.detector.render(
                frames_to_render=0,
                lorentz=True,
                polarization=False,
                structure_factor=False,
                method=method,
            )

            self.assertTrue(torch.sum(torch.isinf(diffraction_pattern)) == 1)
            self.assertTrue(
                torch.sum(diffraction_pattern[~torch.isinf(diffraction_pattern)]) == 0
            )


if __name__ == "__main__":
    unittest.main()
