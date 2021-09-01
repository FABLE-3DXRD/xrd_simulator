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
        # TODO implement this: we need to compute both k and kprime to get the beam geometry
        # and the propagation direction of the diffracted rays. Next we need to compute the
        # element intersections and store these as convex polyhedrons then we can discard the k:s.
        # we want to have a list with "scatterers" which are convex polyhedra with difrfaction vectors
        # in the direction of scattering. The objects also need to specify in what k1,k2 range it scattered
        # as the user will ofcourse not rotate the synchrotron but the sample and thus need to map these things
        # to an affine transformation of the sample stage. 
        pass

    def _get_G(self, U, B, G_hkl):
        return np.dot( U, np.dot( B, G_hkl ) )

    def _get_kprimes( self, k1, k2, U, B, G_hkl, wavelength ):
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
            parametric interval of illumination. If no solutions exist :obj:`None` is returned.
        """

        alpha = self._get_alpha(k1, k2, wavelength)
        assert np.degrees( alpha ) > 1e-6, "The illumination range seems to be fixed, k1 == k2."
        assert np.degrees( alpha ) < 180,  "The illumination must be strictly smaller than 180 dgrs"
        rhat = self._get_rhat(k1, k2)
        G = self._get_G(U, B, G_hkl)
        theta = self._get_bragg_angle( G, wavelength )

        c_0, c_1, c_2 = self._get_tangens_half_angle_equation( k1, theta, G, rhat )
        s1, s2 = self._find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha )

        kprime1, kprime2 = None,None
        if s1 is not None:
            kprime1 = G + self._get_parametric_k( k1, s1, rhat, alpha ) 
        if s2 is not None:
            kprime2 = G + self._get_parametric_k( k1, s2, rhat, alpha )
    
        return kprime1, kprime2

    def _get_alpha(self, k1, k2, wavelength):
        """Compute angle (in radians) between wave vectors k1 and k2.
        """
        return np.arccos( (np.dot( k1, k2 )*wavelength*wavelength)/(4.0*np.pi*np.pi) )

    def _get_rhat(self, k1, k2 ):
        """Compute unit vector normal to the plane holding k1 and k2.
        """
        r = np.cross( k1, k2 )
        return r / np.linalg.norm( r )

    def _get_parametric_k(self, k1, s, rhat, alpha):
        """Finds vector :obj:`k` by rotating :obj:`k1` towards :obj:`k2` a fraction :obj:`s` in the plane spanned by :obj:`k1` and :obj:`k2`. 

        .. math:: \\boldsymbol{k}(s) = \\boldsymbol{k}_i\\cos(s\\alpha)+(\\boldsymbol{\\hat{r}}\\times \\boldsymbol{k}_i)\\sin(s\\alpha)

        Args:
            k1 (:obj:`numpy array`): Intervall start of illuminating wave vectors. (``shape=(3,)``)
            s (:obj:`float`): Angular rotation fraction (s=0 gives k1 and s=1 gives k2 as result).
            rhat (:obj:`numpy array`): unit vector normal to the plane holding k1 and k2. (``shape=(3,)``)
            alpha (:obj:`float`): Angle (in radians) between wave vectors k1 and k2.

        Returns:
            (:obj:`numpy array`): :obj:`k`, the wave vector in the intervall of :obj:`k1` and :obj:`k2`.
        """
        return k1*np.cos( s*alpha ) + np.cross( rhat, k1 )*np.sin( s*alpha )

    def _get_bragg_angle(self, G, wavelength):
        """Compute a Bragg angle given a diffraction (scattering) vector.
        """
        return np.arcsin( np.linalg.norm(G)*wavelength/(4*np.pi) )

    def _get_tangens_half_angle_equation( self, k1, theta, G, rhat ):
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
            theta (:obj:`float`): Bragg angle in radians of sought diffraction.
            G (:obj:`numpy array`): Diffraction (or scattering) vector of sought diffraction. (``shape=(3,)``)

        Returns:
            (:obj:`tuple` of :obj:`float`): Coefficents c_0,c_1 and c_2 of equation (1).
        """
        c_0  = np.dot(G, k1)
        c_1  = np.dot(G, np.cross( rhat, k1 ) )
        c_2  = np.linalg.norm( k1 ) * np.linalg.norm( G ) * np.sin( theta )
        return (c_0, c_1, c_2)

    def _find_solutions_to_tangens_half_angle_equation( self, c_0, c_1, c_2, alpha ):
        """Find all solutions, :obj:`s`, to the equation

        .. math::
            (c_2 - c_0) t^2 + 2 c_1 t + (c_0 + c_2) = 0. \\quad\\quad (1)

        where

        .. math::
            t = \\tan(s \\alpha / 2). \\quad\\quad (2)
        
        and .. math::\\alpha is the angle between k1 and k2

        Args:
            c_0,c_1,c_2 (:obj:`float`): Equation coefficents

        Returns:
            (:obj:`float` or :obj:`None`): solutions if any exists otehrwise returns None.
        """

        if c_0==c_2:
            if c_1==0:
                t1 = t2 = None
            else:
                t1 = -c_0/c_1
        else:
            rootval = (c_1/(c_2 - c_0))**2 -  (c_0 + c_2)/(c_2 - c_0)
            leadingterm = ( -c_1/(c_2 - c_0) ) 
            if rootval<0:
                t1,t2 = None, None
            else:
                t1 = leadingterm + np.sqrt( rootval )
                t2 = leadingterm - np.sqrt( rootval )

        s1, s2 = None, None

        if t1 is not None:
            s1 = 2 * np.arctan( t1 ) / alpha
            if s1>1 or s1<0:
                s1 = None

        if t2 is not None:
            s2 = 2 * np.arctan( t2 ) / alpha
            if s2>1 or s2<0:
                s2 = None


        return s1, s2
        
