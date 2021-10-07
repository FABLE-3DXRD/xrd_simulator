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
        #np.linspace(0, 1, np.degrees( beam.rotator.alpha ).astype(int) )
        #for beam.rotator.alpha
        raise NotImplementedError()

    def diffract(self, beam, detector):
        """Construct the scattering regions in the wavevector range [k1,k2] given a beam profile.

        The beam interacts with the polycrystal producing scattering captured by the detector.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            detector (:obj:`xrd_simulator.detector.Detector`): Object representing a flat rectangular detector.

        """

        # Only Compute the Miller indices that can give diffrraction within the detector bounds
        max_bragg_angle = detector.approximate_wrapping_cone( beam )
        min_bragg_angle = 0

        # TODO: remove this assert and replace by a warning.
        assert max_bragg_angle < 25*np.pi/180, "Maximum Bragg angle is very large, this will result in many hkls..."

        for phase in self.phases:
            phase.set_miller_indices(beam.wavelength, min_bragg_angle, max_bragg_angle) 

        c_1_factor = np.cross( beam.rotator.rhat , beam.k1 )

        scatterers = []

        proximity_intervals = beam.get_proximity_intervals(self.mesh.espherecentroids, self.mesh.eradius)

        for ei in range( self.mesh.number_of_elements ):
            if proximity_intervals[ei][0] is None: continue

            print("Computing for element {} of total elements {}".format(ei,self.mesh.number_of_elements))
            element_vertices = self.mesh.coord[self.mesh.enod[ei]]

            G = laue.get_G(self.eU[ei], self.eB[ei], self.phases[ self.ephase[ei] ].miller_indices.T )
            sinth, normG = laue.get_sin_theta_and_norm_G(G, beam.wavelength)
            c_0s  = np.dot( beam.k1, G)
            c_1s  = np.dot( c_1_factor, G )
            c_2s  = (2 * np.pi / beam.wavelength) * normG * sinth

            for k in range(len(c_0s)):
                for s in laue.find_solutions_to_tangens_half_angle_equation( c_0s[k], c_1s[k], c_2s[k], beam.rotator.alpha ):
                    if s is not None:
                        if utils.contained_by_intervals( s, proximity_intervals[ei] ):
                            beam.set_geometry(s)
                            scattering_region = beam.intersect( element_vertices )
                            if scattering_region is not None:
                                kprime = G[:,k] + beam.k
                                scatterers.append( Scatterer(scattering_region, kprime, s, hkl=None) )  
        beam.set_geometry(s=0)
        detector.frames.append( scatterers )


        # NOTE: Rotations around the beam propagation direction is no true crystal rocking! For a fixed
        # beam in lab frame, the angles between beam and G vectors will not be changed by this! 
        # mathematically: R_beam( v ).dot( beam ) = constant, i.e a dot product cannot change due
        # to rotations around one of the ingoing vectors. Thus we require k1!=k2 for Bragg diffraction
        # to occur.



        
