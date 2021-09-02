import numpy as np 
import matplotlib.pyplot as plt
import utils
from scatterer import Scatterer

class Polycrystal(object):

    """A Polycrystal object links a mesh to a phase-list and hold a ubi matrix to each element
    and can produce corresponding hkl indices and intensity factors for these crystals""" 

    def __init__(self, mesh, ephase, eU, eB, phases ):
        self.mesh   = mesh
        self.ephase = ephase
        self.eU     = eU
        self.eB     = eB
        self.phases = phases

    def get_candidate_elements(self, beam):
        """Get all elements that could diffract for a given illumination setting."""
        pass

    def get_scatterers(self, beam, k1, k2):
        """Construct the scattering regions in the wavevector range [k1,k2] given a beam profile.
        """
        alpha = self._get_alpha(k1, k2)
        # NOTE: Rotations around the beam propagation direction is no true crystal rocking! For a fixed
        # beam in lab frame, the angles between beam and G vectors will not be changed by this! 
        # mathematically: R_beam( v ).dot( beam ) = constant, i.e a dot product cannot change due
        # to rotations around one of the ingoing vectors. Thus we require k1!=k2 for Bragg diffraction
        # to occur.
        assert np.degrees( alpha ) > 1e-6, "The illumination range seems to be fixed, k1 == k2. Rotations around the beam is no crystal rocking!"
        assert np.degrees( alpha ) < 180,  "The illumination must be strictly smaller than 180 dgrs"
        rhat = self._get_rhat(k1, k2)
        rotator = utils.get_planar_rodriguez_rotator(k1, k2)
        scatterers = []
        for ei in range( self.mesh.number_of_elements ):
            scatterers.append( [] )
            # TODO: pass if element not close to beam for speed.
            for G_hkl in self.phases[ self.ephase[ei] ].hkl:
                G = self._get_G(self.eU[ei], self.eB[ei], G_hkl)
                theta = self._get_bragg_angle( G, beam.wavelength )
                c_0, c_1, c_2 = self._get_tangens_half_angle_equation( k1, theta, G, rhat )
                for s in self._find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha ):
                    if s is not None:
                        k      = rotator( k1, s )
                        kprime = G + k
                        beam.align( k )
                        element_vertices = self.mesh.coord[self.mesh.enod[ei]]
                        hs = beam.intersect( element_vertices )
                        scatterers.append( Scatterer(hs, kprime) )        
        return scatterers

    def _get_G(self, U, B, G_hkl):
        return np.dot( U, np.dot( B, G_hkl ) )

    def _get_alpha(self, k1, k2, wavelength):
        """Compute angle (in radians) between wave vectors k1 and k2.
        """
        return np.arccos( (np.dot( k1, k2 )*wavelength*wavelength)/(4.0*np.pi*np.pi) )

    def _get_rhat(self, k1, k2 ):
        """Compute unit vector normal to the plane holding k1 and k2.
        """
        r = np.cross( k1, k2 )
        return r / np.linalg.norm( r )

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
        
