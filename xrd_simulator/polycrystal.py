import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class Polycrystal(object):

    """A Polycrystal object links a mesh to a phase-list and hold a ubi matrix to each element
    and can produce corresponding hkl indices and intensity factors for these crystals""" 

    def __init__(self, mesh, ephase, phases ):
        self.mesh   = mesh
        self.ephase = ephase
        self.phases = phases

    def get_crystal_beam_intersections(self, beam, elements):
        """Compute convex intersection hulls of beam and selected mesh elements."""
        pass

    def get_candidate_elements(self, beam):
        """Get all elements that could diffract for a given illumination setting."""
        pass

    def get_scattered_rays(self, beam):
        """Construct ScatteredRay objects for each scattering occurence and return."""
        pass

    def get_kprimes( self, k1, k2, U, B, G_hkl, wavelength ):
        """Check for difraction from a crystal for a specific hkl family and illumination intervall, [k1,k2].

        The intervall is defined by rotating the illuminating wave vector k1 into k2 in the plane spanned 
        by k1 and k2. The crystal is defined by its U and B matrix. 

        Args:
            k1 (:obj:`numpy array`): Intervall start of illuminating wave vectors. (``shape=(3,)``)
            k2 (:obj:`numpy array`): Intervall end of illuminating wave vectors. (``shape=(3,)``)
            U (:obj:`numpy array`): Crystal coordinates to sample coordinates mapping matrix. (``shape=(3,3)``)
            B (:obj:`numpy array`): Crystal reciprocal coordinates to real coordinates mapping matrix. (``shape=(3,3)``)
            G_hkl (:obj:`numpy array`): Diffraction (or scattering) Miller indices. (``shape=(3,)``)
            wavelength (:obj:`float`): X-ray wavelength.

        Returns:
            (:obj:`numpy array` or :obj:`None`): Solutions to diffraction equations for the given crystal and the given
            parametric intervbal of illumination. If no solutions exist :obj:`None` is returned.
        """
        assert np.degrees( np.arccos( np.dot(k1, k2) / ( np.linalg.norm(k1) * np.linalg.norm(k2) ) ) ) > 1e-6 # Require k1 and k2 to be distinct.
        G = np.dot( U, np.dot( B, G_hkl ) )
        theta = self.get_bragg_angle( G, wavelength )
        c_1,c_2,c_2 = self.get_tangens_half_angle_equation( k1, k2, theta, G )
        s1,s2 = self.find_solutions_to_tangens_half_angle_equation( k1, k2, c_1, c_2, c_2 )

        if s1 is None:
            kprime1 = None
        else:
            kprime1 = G + self.get_parametric_k( k1, k2, s1 )
        
        if s2 is None:
            kprime2 = None
        else:
            kprime2 = G + self.get_parametric_k( k1, k2, s2 )

        return kprime1, kprime2

    def get_parametric_k(self, k1, k2, s):
        """Finds vector :obj:`k` by rotating :obj:`k1` towards :obj:`k2` a fraction :obj:`s` in the plane spanned by :obj:`k1` and :obj:`k2`. 

        .. math:: \\boldsymbol{k}(s) = \\boldsymbol{k}_i\\cos(s\\alpha)+(\\boldsymbol{\\hat{r}}\\times \\boldsymbol{k}_i)\\sin(s\\alpha)

        Args:
            k1 (:obj:`numpy array`): Intervall start of illuminating wave vectors. (``shape=(3,)``)
            k2 (:obj:`numpy array`): Intervall end of illuminating wave vectors. (``shape=(3,)``)
            s (:obj:`float`): Angular rotation fraction (s=0 gives k1 and s=1 gives k2 as result).

        Returns:
            (:obj:`numpy array`): :obj:`k`, the wave vector in the intervall of :obj:`k1` and :obj:`k2`.
        """
        r     = np.cross( k1, k2 )
        rhat  = r / np.linalg.norm( r )
        alpha = np.arccos( np.dot(k1, k2) / ( np.linalg.norm(k1) * np.linalg.norm(k2) ) )
        return k1*np.cos(s*alpha) + np.cross( rhat, k1 )*np.sin( s*alpha )

    def get_bragg_angle(self, G, wavelength):
        """Compute a Bragg angle given a diffraction (scattering) vector.
        """
        return np.arcsin( np.linalg.norm(G)*wavelength/(4*np.pi) )

    def get_tangens_half_angle_equation( self, k1, k2, theta, G ):
        """Find the coefficents of the quadratic equation 

        .. math::
            (c_2 - c_0) t^2 + 2 c_1 t + (c_0 + c_2) = 0. \\quad\\quad (1)
            
        Its roots (if existing) are solutions to the Laue equations where t is

        .. math::
            t = \\tan(s \\alpha / 2). \\quad\\quad (2)
        
        and s is the sought parameter which parametrises all wave vectors between k1 and k2 where 
        s=0 corresponds to k1 and s=1 to k2. Intermediate vectors are reached by rotating k1 in the
        plane spanned by k1 and k2.

        Args:
            k1 (:obj:`numpy array`): Intervall start of illuminating wave vectors. (``shape=(3,)``)
            k2 (:obj:`numpy array`): Intervall end of illuminating wave vectors. (``shape=(3,)``)
            theta (:obj:`float`): Bragg angle in radians of sought diffraction.
            G (:obj:`numpy array`): Diffraction (or scattering) vector of sought diffraction. (``shape=(3,)``)

        Returns:
            (:obj:`tuple` of :obj:`float`): Coefficents c_0,c_1 and c_2 of equation (1).
        """
        r    = np.cross( k1, k2 )
        rhat = r / np.linalg.norm( r ) 
        c_0  = np.dot(G, k1)
        c_1  = np.dot(G, np.cross( rhat, k1 ) )
        c_2  = - np.linalg.norm( k1 ) * np.linalg.norm( G ) * np.sin( theta )
        return (c_0, c_1, c_2)

    def find_solutions_to_tangens_half_angle_equation( self, k1, k2, c_0, c_1, c_2 ):
        """Find all solutions, s, to the equation

        .. math::
            (c_2 - c_0) t^2 + 2 c_1 t + (c_0 + c_2) = 0. \\quad\\quad (1)

        where

        .. math::
            t = \\tan(s \\alpha / 2). \\quad\\quad (2)
        
        and .. math::\\alpha is the angle between k1 and k2

        Args:
            k1 (:obj:`numpy array`): Intervall start of illuminating wave vectors. (``shape=(3,)``)
            k2 (:obj:`numpy array`): Intervall end of illuminating wave vectors. (``shape=(3,)``)
            c_0,c_1,c_2 (:obj:`float`): Equation coefficents

        Returns:
            (:obj:`float` or :obj:`None`): solutions (s1, s2) if any exists otehrwise returns (None, None).
        """

        alpha = np.arccos( np.dot(k1, k2) / ( np.linalg.norm(k1) * np.linalg.norm(k2) ) )

        if c_0==c_2:
            if c_1==0:
                t1 = t2 = None
            else:
                t1 = -c_0/c_1
        else:
            t1 = ( -c_1/(c_2 - c_0) ) + np.sqrt( (c_1/(c_2 - c_0))**2 -  (c_0 + c_2)/(c_2 - c_0) )
            if np.imag(t1)!=0:
                t1 = None
            t2 = ( -c_1/(c_2 - c_0) ) - np.sqrt( (c_1/(c_2 - c_0))**2 -  (c_0 + c_2)/(c_2 - c_0) )
            if np.imag(t2)!=0:
                t2 = None

        if t1 is not None:
            s1 = 2 * np.arctan( t1 ) / alpha
            if s1>1 or s1<0:
                s1 = None
        else:
            s1 = None

        if t2 is not None:
            s2 = 2 * np.arctan( t2 ) / alpha
            if s1>1 or s1<0:
                s1 = None
        else:
            s2 = None

        return s1, s2
