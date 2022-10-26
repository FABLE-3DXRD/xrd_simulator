import unittest
import numpy as np
from xfab import tools
from xrd_simulator import laue, utils


class TestLaue(unittest.TestCase):

    """Unit test for the laue.py functions module.
    """

    def setUp(self):
        np.random.seed(10)  # changes all randomisation in the test

    def test_get_G_and_bragg_angle(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = self.get_pseudorandom_wavelength()

        G = laue.get_G(U, B, G_hkl=np.array([[1, 2, -1], [1, 3, -1]]).T)
        theta = laue.get_bragg_angle(G, wavelength)
        d = 2 * np.pi / np.linalg.norm(G, axis=0)

        for i in range(len(theta)):
            self.assertLessEqual(np.linalg.norm(
                G[:, i]), 4 * np.pi / wavelength, msg="Length of G is wrong")
            self.assertLessEqual(theta[i], np.pi, msg="Bragg angle too large")
            self.assertGreaterEqual(theta[i], 0, msg="Bragg angle is negative")
            self.assertAlmostEqual(
                np.sin(
                    theta[i]) * 2 * d[i],
                wavelength,
                msg="G and theta does not fulfill Braggs law")

    def test_get_sin_theta_and_norm_G(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = self.get_pseudorandom_wavelength()
        G = laue.get_G(U, B, G_hkl=np.array([[1, 2, -1], [1, 3, -1]]).T)
        theta = laue.get_bragg_angle(G, wavelength)

        sinth, Gnorm = laue.get_sin_theta_and_norm_G(G, wavelength)

        for i in range(len(theta)):
            self.assertAlmostEqual(
                sinth[i], np.sin(
                    theta[i]), msg="error theta")
            self.assertAlmostEqual(Gnorm[i], np.linalg.norm(
                G[:, i]), msg="error in norm of G")

    def test_find_solutions_to_tangens_half_angle_equation(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = cell[0] / \
            18.  # make sure the wavelength fits in the lattice spacing
        # select a large interval of k-vectors
        k = np.array([1.0, 0, 0]) * 2 * np.pi / wavelength
        rotation_axis = np.random.rand(3,)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.pi / 2.
        # let the crystal be aligned to assure hkl=100 to be in its Bragg condition
        # for the k-intervall
        U = np.eye(3, 3)
        G_0 = laue.get_G(U, B, G_hkl=np.array([-1, 0, 0])).reshape(3,1)

        rx, ry, rz = rotation_axis
        K = np.array([[0, -rz, ry],
                        [rz, 0, -rx],
                        [-ry, rx, 0]])
        rho_0 = -k.dot( K.dot(K) ).dot(G_0)
        rho_1 =  k.dot( K ).dot(G_0)
        rho_2 =  k.dot( np.eye(3, 3) + K.dot(K) ).dot(G_0) + np.sum((G_0 * G_0), axis=0) / 2.
        t1s, t2s = laue.find_solutions_to_tangens_half_angle_equation(
            rho_0, rho_1, rho_2, rotation_angle)

        for i,(t1,t2) in enumerate(zip(t1s,t2s)):
            # Check that at least one solution has been found and that it satisfies
            # the half angle equation.
            self.assertTrue((~np.isnan(t1) or ~np.isnan(t2)),
                            msg="Tangens half angle equation could not be solved")
            if ~np.isnan(t1):
                self.assertLessEqual(t1, 1, msg="s>1")
                self.assertGreaterEqual(t1, 0, msg="s>1")
                t1 = np.tan(t1 * rotation_angle / 2.)
                self.assertAlmostEqual((rho_2[i] - rho_0[i]) * t1**2 + 2 * rho_1[i] *
                                    t1 + (rho_0[i] + rho_2[i]), 0, msg="Parametric solution wrong")
            if ~np.isnan(t2):
                self.assertLessEqual(t2, 1, msg="s>1")
                self.assertGreaterEqual(t2, 0, msg="s<0")
                t2 = np.tan(t1 * rotation_angle / 2.)
                self.assertAlmostEqual((rho_2[i] - rho_0[i]) * t2**2 + 2 * rho_1[i] *
                                    t2 + (rho_0[i] + rho_2[i]), 0, msg="Parametric solution wrong")

    def get_pseudorandom_crystal(self):
        phi1, PHI, phi2 = np.random.rand(3,) * 2 * np.pi
        U = tools.euler_to_u(phi1, PHI, phi2)
        strain_tensor = (np.random.rand(6,) - 0.5) / 50.
        unit_cell = [
            2 + np.random.rand() * 5,
            2 + np.random.rand() * 5,
            2 + np.random.rand() * 5,
            90.,
            90.,
            90.]
        B = utils._epsilon_to_b(strain_tensor, unit_cell)
        return U, B, unit_cell, strain_tensor

    def get_pseudorandom_wavelength(self):
        return np.random.rand() * 0.5

    def get_pseudorandom_wave_vector(self, wavelength):
        k = (np.random.rand(3,) - 0.5) * 2
        k = 2 * np.pi * k / (np.linalg.norm(k) * wavelength)
        return k


if __name__ == '__main__':
    unittest.main()
