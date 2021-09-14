import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.utils import source
from xrd_simulator.scatterer import Scatterer
from xrd_simulator import utils
from xrd_simulator import laue

class Polycrystal(object):

    """Represents a multi-phase polycrystal as a tetrahedral mesh where each element can be a single crystal
    
    This object is arguably the most complex entity in the package as it links a several phase objects to a mesh
    and lets them interact with a beam and detector object.

    Args:
        mesh (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the 
            geometry of the sample.
        ephase (:obj:`numpy array`): Index of phase that elements belong to such that phases[ephase[i]] gives the
            xrd_simulator.phase.Phase object of element number i.
        eU (:obj:`numpy array`): Per element U (orinetation) matrices, (```shape=(N,3,3)```).
        eB (:obj:`numpy array`): Per element B (hkl to crystal mapper) matrices, (```shape=(N,3,3)```).
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.

    Attributes:
        mesh (:obj:`xrd_simulator.mesh.TetraMesh`): Object representing a tetrahedral mesh which defines the 
            geometry of the sample.
        ephase (:obj:`numpy array`): Index of phase that elements belong to such that phases[ephase[i]] gives the
            xrd_simulator.phase.Phase object of element number i.
        eU (:obj:`numpy array`): Per element U (orinetation) matrices, (```shape=(N,3,3)```).
        eB (:obj:`numpy array`): Per element B (hkl to crystal mapper) matrices, (```shape=(N,3,3)```).
        phases (:obj:`list` of :obj:`xrd_simulator.phase.Phase`): List of all unique phases present in the polycrystal.

    """ 

    def __init__(self, mesh, ephase, eU, eB, phases ):
        self.mesh   = mesh
        self.ephase = ephase
        self.eU     = eU
        self.eB     = eB
        self.phases = phases

    def get_candidate_elements(self, beam):
        """Get all elements that could diffract for a given illumination setting.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            
        Returns:
        
        """
        raise NotImplementedError()

    def diffract(self, beam, detector):
        """Construct the scattering regions in the wavevector range [k1,k2] given a beam profile.

        The beam interacts with the polycrystal producing scattering captured by the detector.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            detector (:obj:`xrd_simulator.detector.Detector`): Object representing a flat rectangular detector.

        """

        rotator = utils.RodriguezRotator(beam.k1, beam.k2)


        # Only Compute the Miller indices that can give diffrraction within the detector bounds
        max_bragg_angle = detector.approximate_wrapping_cone( beam, source_point=self.mesh.centroid )
        min_bragg_angle = 0

        hkls = [phase.generate_miller_indices(beam.wavelength, min_bragg_angle, max_bragg_angle) for phase in self.phases]

        scatterers = []

        for ei in range( self.mesh.number_of_elements ):
            element_vertices = self.mesh.coord[self.mesh.enod[ei]]
            # TODO: pass if element not close to beam for speed.
            print(max_bragg_angle, hkls[ self.ephase[ei] ].shape, self.mesh.number_of_elements)
            for G_hkl in hkls[ self.ephase[ei] ]:
                G = laue.get_G(self.eU[ei], self.eB[ei], G_hkl)
                theta = laue.get_bragg_angle( G, beam.wavelength )
                c_0, c_1, c_2 = laue.get_tangens_half_angle_equation( beam.k1, theta, G, rotator.rhat )
                for s in laue.find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, rotator.alpha ):
                    if s is not None:
                        beam.set_geometry(s)
                        hs = beam.intersect( element_vertices )
                        if hs is not None:
                            kprime = G + beam.k
                            scatterers.append( Scatterer(hs, kprime, s) )  
        beam.set_geometry(s=0)
        detector.frames.append( scatterers )


        # NOTE: Rotations around the beam propagation direction is no true crystal rocking! For a fixed
        # beam in lab frame, the angles between beam and G vectors will not be changed by this! 
        # mathematically: R_beam( v ).dot( beam ) = constant, i.e a dot product cannot change due
        # to rotations around one of the ingoing vectors. Thus we require k1!=k2 for Bragg diffraction
        # to occur.



        
