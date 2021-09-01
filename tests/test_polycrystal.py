import unittest
import numpy as np
from xfab import tools
from xrd_simulator.polycrystal import Polycrystal

class TestPolycrystal(unittest.TestCase):

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test

    def test_init(self):
        pc = Polycrystal(None, None, None)

    def test_get_G_and_bragg_angle(self):
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = self.get_pseudorandom_wavelength()
        pc = Polycrystal(None, None, None)
        G = pc._get_G(U, B, G_hkl=np.array([1, 2, -1]))
        theta = pc._get_bragg_angle(G, wavelength)
        d = 2*np.pi / np.linalg.norm(G)

        self.assertLessEqual( np.linalg.norm(G), 4*np.pi / wavelength, msg="Length of G is wrong" )
        self.assertLessEqual( theta, np.pi, msg="Bragg angle too large" )
        self.assertGreaterEqual( theta, 0, msg="Bragg angle is negative" )
        self.assertAlmostEqual( np.sin(theta)*2*d, wavelength, msg="G and theta does not fulfill Braggs law" )

    def test_get_rhat(self):
        pc = Polycrystal(None, None, None)
        wavelength = self.get_pseudorandom_wavelength()
        k1, k2 = self.get_pseudorandom_k1_k2(wavelength)
        rhat = pc._get_rhat(k1,k2)

        self.assertAlmostEqual( rhat.dot(k1), 0, msg="rhat is not orthogonal to k1" )
        self.assertAlmostEqual( rhat.dot(k2), 0, msg="rhat is not orthogonal to k2" )
        self.assertAlmostEqual( np.linalg.norm(rhat), 1.0, msg="rhat is not normalised" )

    def test_get_alpha(self):
        pc = Polycrystal(None, None, None)
        wavelength = self.get_pseudorandom_wavelength()
        k1, k2 = self.get_pseudorandom_k1_k2(wavelength)
        alpha = pc._get_alpha(k1 ,k2, wavelength)
        k1dotk2 = np.cos(alpha)*4*np.pi*np.pi/(wavelength*wavelength)

        self.assertLessEqual( alpha, np.pi, msg="alpha is greater than pi" )
        self.assertGreaterEqual( alpha, 0, msg="alpha is negative" )
        self.assertAlmostEqual( k1dotk2, k1.dot(k2), msg="alpha does not match dot product cosine formula" )

    def test_get_parametric_k(self):
        pc = Polycrystal(None, None, None)
        wavelength = self.get_pseudorandom_wavelength()
        k1, k2     = self.get_pseudorandom_k1_k2(wavelength)
        s = np.random.rand()
        alpha   = pc._get_alpha(k1 ,k2, wavelength)
        rhat    = pc._get_rhat(k1,k2)
        k       = pc._get_parametric_k( k1, s, rhat, alpha)
        k1dotk2 = np.cos(alpha)*4*np.pi*np.pi/(wavelength*wavelength)

        kdotk1 = np.cos(s*alpha) * (2*np.pi/wavelength)**2
        kdotk2 = np.cos((1-s)*alpha) * (2*np.pi/wavelength)**2

        self.assertAlmostEqual( np.linalg.norm(k), 2*np.pi/wavelength, msg="Wavevector length is wrong" )
        self.assertAlmostEqual( k.dot(rhat), 0., msg="k not in the plane of k1 and k2" )
        self.assertAlmostEqual( kdotk1, k.dot(k1), msg="cosine formula mismatch in k * k1" )
        self.assertAlmostEqual( kdotk2, k.dot(k2), msg="cosine formula mismatch in k * k2" )

    def test_get_kprimes(self):
        pc = Polycrystal(None, None, None)
        _, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = cell[0]/10. # make sure the wavelength fits in the lattice spacing
        k1 = np.array([ 1.0, 0, 0 ])*2*np.pi/wavelength # select a large interval of k-vectors
        k2 = np.array([ 0.0, 1.0, 0 ])*2*np.pi/wavelength
        G_hkl = np.array([-1, 0, 0]) 
        U = np.eye(3,3) # let the crystal be aligned to assure 100 to be in its Bragg condition for the k-intervall
        kprime1, kprime2 = pc._get_kprimes( k1, k2, U, B, G_hkl, wavelength )

        alpha   = pc._get_alpha(k1 ,k2, wavelength)
        rhat    = pc._get_rhat(k1,k2)
        G  = pc._get_G(U, B, G_hkl)
        theta   = pc._get_bragg_angle(G, wavelength)
        c_0, c_1, c_2 = pc._get_tangens_half_angle_equation(k1, theta, G, rhat ) 
        s1, s2 = pc._find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha )

        for kprime,s,name in zip([kprime1,kprime2],[s1,s2],["kprime1","kprime2"]):
            if kprime is not None:
                self.assertAlmostEqual( np.linalg.norm(kprime), 2*np.pi/wavelength, msg="L2 norm of "+name+" is not 2*pi/wavelength" )
                bragg_angle = -np.arccos( np.dot( kprime/(2*np.pi/wavelength), G/np.linalg.norm(G)) ) + np.pi/2. 
                self.assertAlmostEqual( bragg_angle, theta, msg=name+" does not form bragg angle to planes with noraml in G direction" )
                k  = pc._get_parametric_k(k1, s, rhat, alpha)
                self.assertAlmostEqual( np.linalg.norm( kprime-k-G ), 0 , msg="Scattering equation definition not fulfilled; "+name+" - k != G " )

    def test_get_tangens_half_angle_equation(self):
        pc = Polycrystal(None, None, None)
        wavelength = self.get_pseudorandom_wavelength()
        U, B, cell, strain = self.get_pseudorandom_crystal()
        k1, k2     = self.get_pseudorandom_k1_k2(wavelength)
        rhat = pc._get_rhat(k1, k2)
        G = pc._get_G(U, B, G_hkl=np.array([-1, 0, 0]))
        theta  = pc._get_bragg_angle( G, wavelength )
        c_0, c_1, c_2 = pc._get_tangens_half_angle_equation(k1, theta, G, rhat ) 
        self.assertTrue( np.isreal(c_0) )
        self.assertTrue( np.isreal(c_1) )
        self.assertTrue( np.isreal(c_2) )

    def test_find_solutions_to_tangens_half_angle_equation(self):
        pc = Polycrystal(None, None, None)
        U, B, cell, strain = self.get_pseudorandom_crystal()
        wavelength = cell[0]/18. # make sure the wavelength fits in the lattice spacing
        k1 = np.array([ 1.0, 0, 0 ])*2*np.pi/wavelength # select a large interval of k-vectors
        k2 = np.array([ 0.0, 1.0, 0 ])*2*np.pi/wavelength
        U = np.eye(3,3) # let the crystal be aligned to assure 100 to be in its Bragg condition for the k-intervall
        rhat = pc._get_rhat(k1, k2)
        G = pc._get_G(U, B, G_hkl=np.array([-1, 0, 0]))
        theta  = pc._get_bragg_angle( G, wavelength )
        alpha  = pc._get_alpha(k1 ,k2, wavelength)
        c_0, c_1, c_2 = pc._get_tangens_half_angle_equation(k1, theta, G, rhat ) 
        s1, s2 = pc._find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha )

        # Check that at least one solution has been found and that it satisfies the half angle equation.
        self.assertTrue( ( (s1 is not None) or (s2 is not None) ), msg="Tangens half angle equation could not be solved")
        if s1 is not None:
            t1 = np.tan( s1*alpha/2. )
            self.assertAlmostEqual( (c_2 - c_0)*t1**2 + 2*c_1*t1 + (c_0 + c_2), 0, msg="Parametric solution wrong")
        if s2 is not None:
            t2 = np.tan( s1*alpha/2. )
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