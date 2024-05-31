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

        G = laue.get_G(U, B, G_hkl=np.array([[1, 2, -1], [1, 3, -1]]))
        theta = laue.get_bragg_angle(G, wavelength)
        d = 2 * np.pi / np.linalg.norm(G, axis=0)

        for i,angle in enumerate(theta):
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
        G = laue.get_G(U, B, G_hkl=np.array([[1, 2, -1], [1, 3, -1]]))
        theta = laue.get_bragg_angle(G, wavelength)
        sinth, Gnorm = laue.get_sin_theta_and_norm_G(G, wavelength)
        for i,angles in enumerate(theta):
            self.assertAlmostEqual(
                sinth[i], np.sin(
                    theta[i]), msg="error theta")
            self.assertAlmostEqual(Gnorm[i], np.linalg.norm(G[:, i]), msg="error in norm of G",places=6)

    def test_find_solutions_to_tangens_half_angle_equation(self):
        """_Test to check if find_solutions_to_tangens_half_angle equation works properly_
        """
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
        rho_0_factor = -k.dot( K.dot(K) ) # The operations involving G_0 are moved inside find_solutions_to_tangens...
        rho_1_factor =  k.dot( K ) # The operations involving G_0 are moved inside find_solutions_to_tangens...
        rho_2_factor =  k.dot( np.eye(3, 3) + K.dot(K) )  # The operations involving G_0 are moved inside find_solutions_to_tangens...
        


        # Now G_0 and rho_factors are sent before computation to save memory when diffracting many grains.
        reflection_index, time_values = laue.find_solutions_to_tangens_half_angle_equation(G_0,rho_0_factor, rho_1_factor, rho_2_factor, rotation_angle)
        
        G_0 = G_0[np.newaxis,:,:]
        rho_0 = np.matmul(rho_0_factor,G_0)
        rho_2 = np.matmul(rho_2_factor,G_0)+ np.sum((G_0 * G_0), axis=1) / 2.   
        rho_1 = np.matmul(rho_1_factor,G_0)
        
        for t in time_values:
            # Check that at least one solution has been found and that it satisfies
            # the half angle equation.
            self.assertTrue((~np.isnan(t)),
                            msg="Tangens half angle equation could not be solved")
            if ~np.isnan(t):
                self.assertLessEqual(t, 1, msg="s>1")
                self.assertGreaterEqual(t, 0)
                t = np.tan(t * rotation_angle / 2.)
                equation = (rho_2 - rho_0) * t**2 + 2 * rho_1 *t + (rho_0 + rho_2)
                self.assertAlmostEqual(equation.item(), 0, msg="Parametric solution wrong")


    def get_pseudorandom_crystal(self):
        phi1, PHI, phi2 = np.random.rand(3,) * 2 * np.pi
        U = tools.euler_to_u(phi1, PHI, phi2)
        epsilon = (np.random.rand(6,) - 0.5) / 50.
        unit_cell = [
            2 + np.random.rand() * 5,
            2 + np.random.rand() * 5,
            2 + np.random.rand() * 5,
            90.,
            90.,
            90.]
        
        strain_tensor = utils._strain_as_tensor(epsilon)
        B0 = tools.form_b_mat(unit_cell)
        B = utils._epsilon_to_b(strain_tensor, B0)
        return U, B, unit_cell, strain_tensor

    def get_pseudorandom_wavelength(self):
        return np.random.rand() * 0.5

    def get_pseudorandom_wave_vector(self, wavelength):
        k = (np.random.rand(3,) - 0.5) * 2
        k = 2 * np.pi * k / (np.linalg.norm(k) * wavelength)
        return k


if __name__ == '__main__':
    unittest.main()
