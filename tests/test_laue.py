import unittest
import numpy as np
from xfab import tools
from xrd_simulator import laue
from xrd_simulator import utils

class TestLaue(unittest.TestCase):

    """Unit test for the laue.py functions module. 
    """

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test

    def test_get_G_and_bragg_angle(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = self.get_pseudorandom_wavelength()
        
        G = laue.get_G(U, B, G_hkl=np.array([[1, 2, -1], [1, 3, -1]]).T)
        theta = laue.get_bragg_angle(G, wavelength)
        d = 2*np.pi / np.linalg.norm(G, axis=0)

        for i in range(len(theta)):        
            self.assertLessEqual( np.linalg.norm(G[:,i]), 4*np.pi / wavelength, msg="Length of G is wrong" )
            self.assertLessEqual( theta[i], np.pi, msg="Bragg angle too large" )
            self.assertGreaterEqual( theta[i], 0, msg="Bragg angle is negative" )
            self.assertAlmostEqual( np.sin(theta[i])*2*d[i], wavelength, msg="G and theta does not fulfill Braggs law" )

    def test_get_sin_theta_and_norm_G(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = self.get_pseudorandom_wavelength()
        G = laue.get_G(U, B, G_hkl=np.array([[1, 2, -1], [1, 3, -1]]).T)
        theta = laue.get_bragg_angle(G, wavelength)

        sinth, Gnorm = laue.get_sin_theta_and_norm_G( G, wavelength )

        for i in range(len(theta)):
            self.assertAlmostEqual(sinth[i], np.sin(theta[i]), msg="error theta")
            self.assertAlmostEqual(Gnorm[i], np.linalg.norm(G[:,i]), msg="error in norm of G")

    def test_get_tangens_half_angle_equation(self):
        wavelength = self.get_pseudorandom_wavelength()
        U, B, cell, strain = self.get_pseudorandom_crystal()
        k1, k2     = self.get_pseudorandom_k1_k2(wavelength)
        rotator    = utils.RodriguezRotator( k1, k2 )
        G = laue.get_G(U, B, G_hkl=np.array([-1, 0, 0]))
        theta  = laue.get_bragg_angle( G, wavelength )
        c_0, c_1, c_2 = laue.get_tangens_half_angle_equation(k1, theta, G, rotator.rhat ) 
        self.assertTrue( np.isreal(c_0) )
        self.assertTrue( np.isreal(c_1) )
        self.assertTrue( np.isreal(c_2) )

    def test_find_solutions_to_tangens_half_angle_equation(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = cell[0]/18. # make sure the wavelength fits in the lattice spacing
        k1 = np.array([ 1.0, 0, 0 ])*2*np.pi/wavelength # select a large interval of k-vectors
        k2 = np.array([ 0.0, 1.0, 0 ])*2*np.pi/wavelength
        U = np.eye(3,3) # let the crystal be aligned to assure 100 to be in its Bragg condition for the k-intervall
        rotator    = utils.RodriguezRotator( k1, k2 )
        G = laue.get_G(U, B, G_hkl=np.array([-1, 0, 0]))
        theta  = laue.get_bragg_angle( G, wavelength )
        c_0, c_1, c_2 = laue.get_tangens_half_angle_equation(k1, theta, G, rotator.rhat ) 
        s1, s2 = laue.find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, rotator.alpha )

        # Check that at least one solution has been found and that it satisfies the half angle equation.
        self.assertTrue( ( (s1 is not None) or (s2 is not None) ), msg="Tangens half angle equation could not be solved")
        if s1 is not None:
            self.assertLessEqual( s1, 1, msg="s>1")
            self.assertGreaterEqual( s1, 0, msg="s>1")
            t1 = np.tan( s1*rotator.alpha/2. )
            self.assertAlmostEqual( (c_2 - c_0)*t1**2 + 2*c_1*t1 + (c_0 + c_2), 0, msg="Parametric solution wrong")
        if s2 is not None:
            self.assertLessEqual( s2, 1, msg="s>1")
            self.assertGreaterEqual( s2, 0, msg="s<0")
            t2 = np.tan( s1*rotator.alpha/2. )
            self.assertAlmostEqual( (c_2 - c_0)*t2**2 + 2*c_1*t2 + (c_0 + c_2), 0, msg="Parametric solution wrong")

    def get_pseudorandom_crystal(self):
        phi1, PHI, phi2 = np.random.rand(3,)*2*np.pi
        U = tools.euler_to_u(phi1, PHI, phi2)
        strain_tensor = (np.random.rand(6,)-0.5)/50.
        unit_cell = [2+np.random.rand()*5, 2+np.random.rand()*5, 2+np.random.rand()*5, 90., 90., 90.]
        B = tools.epsilon_to_b( strain_tensor, unit_cell )
        return U, B, unit_cell, strain_tensor

    def get_pseudorandom_wavelength(self):
        return np.random.rand()*0.5

    def get_pseudorandom_k1_k2(self, wavelength):
        k1 = (np.random.rand(3,)-0.5)*2
        k2 = (np.random.rand(3,)-0.5)*2
        k1 = 2*np.pi*k1/(np.linalg.norm(k1)*wavelength)
        k2 = 2*np.pi*k2/(np.linalg.norm(k2)*wavelength)
        return k1, k2

if __name__ == '__main__':
    unittest.main()