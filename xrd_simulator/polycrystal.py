import numpy as np 
import matplotlib.pyplot as plt
from xrd_simulator.scatterer import Scatterer
from xrd_simulator import utils
from xrd_simulator import laue

class Polycrystal(object):

    """A Polycrystal object links a mesh to a phase-list and hold a U and B matrix to each element
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

    def diffract(self, beam, detector):
        """Construct the scattering regions in the wavevector range [k1,k2] given a beam profile.
        """
        alpha = lau.get_alpha(k1, k2)
        assert np.degrees( alpha ) > 1e-6, "The illumination range seems to be fixed, k1 == k2. Rotations around the beam is no crystal rocking!"
        assert np.degrees( alpha ) < 180,  "The illumination must be strictly smaller than 180 dgrs"

        # NOTE: Rotations around the beam propagation direction is no true crystal rocking! For a fixed
        # beam in lab frame, the angles between beam and G vectors will not be changed by this! 
        # mathematically: R_beam( v ).dot( beam ) = constant, i.e a dot product cannot change due
        # to rotations around one of the ingoing vectors. Thus we require k1!=k2 for Bragg diffraction
        # to occur.

        rhat = lau.get_rhat(k1, k2)
        scatterers = []

        # Compute the Miller indices based on the rotational intervall.
        max_bragg_angle = approximate_wrapping_cone( beam )
        min_bragg_angle = 0
        hkls = [phase.generate_miller_indices(wavelength, min_bragg_angle, max_bragg_angle) for phase in self.phases]

        for ei in range( self.mesh.number_of_elements ):
            scatterers.append( [] )
            # TODO: pass if element not close to beam for speed.
            for G_hkl in hkls[ self.ephase[ei] ]:
                G = lau.get_G(self.eU[ei], self.eB[ei], G_hkl)
                theta = lau.get_bragg_angle( G, beam.wavelength )
                c_0, c_1, c_2 = lau.get_tangens_half_angle_equation( beam.k1, theta, G, rhat )
                for s in self._find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha ):
                    if s is not None:
                        beam.set_geometry(s)
                        kprime = G + beam.k
                        element_vertices = self.mesh.coord[self.mesh.enod[ei]]
                        hs = beam.intersect( element_vertices )
                        scatterers.append( Scatterer(hs, kprime, s) )  
        beam.set_geometry(s=0)
        detector.frames.append( scatterers )


        
