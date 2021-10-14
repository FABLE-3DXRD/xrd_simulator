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

    def diffract(self, beam, detector, rigid_body_motion, min_bragg_angle=0, max_bragg_angle=None):
        """Construct the scattering regions in the wavevector range [k1,k2] given a beam profile.

        The beam interacts with the polycrystal producing scattering captured by the detector.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            detector (:obj:`xrd_simulator.detector.Detector`): Object representing a flat rectangular detector.

        """

        # Only Compute the Miller indices that can give diffrraction within the detector bounds
        if max_bragg_angle is None:
            max_bragg_angle = detector.get_wrapping_cone( beam.wavevector, beam.centroid )

        # TODO: remove this assert and replace by a warning.
        assert max_bragg_angle < 25*np.pi/180, "Maximum Bragg angle is very large, this will result in many hkls..."

        for phase in self.phases:
            phase.setup_diffracting_planes(beam.wavelength, min_bragg_angle, max_bragg_angle) 

        c_0_factor   = -beam.wavevector.T.dot( rigid_body_motion.rotator.K2 )
        c_1_factor   =  beam.wavevector.T.dot( rigid_body_motion.rotator.K )
        c_2_factor   =  beam.wavevector.T.dot( rigid_body_motion.rotator.I + rigid_body_motion.rotator.K2 )

        scatterers = []

        proximity_intervals = beam.get_proximity_intervals(self.mesh.espherecentroids, self.mesh.eradius, rigid_body_motion) #TODO: fix this 

        for ei in range( self.mesh.number_of_elements ):
            if proximity_intervals[ei][0] is None: continue

            print("Computing for element {} of total elements {}".format(ei,self.mesh.number_of_elements))
            element_vertices_0 = self.mesh.coord[self.mesh.enod[ei]] 
            G_0 = laue.get_G(self.eU[ei], self.eB[ei], self.phases[ self.ephase[ei] ].miller_indices.T )

            c_0s  = c_0_factor.dot(G_0)
            c_1s  = c_1_factor.dot(G_0)
            sinth, normG = laue.get_sin_theta_and_norm_G(G_0, beam.wavelength)
            c_2s  = c_2_factor.dot(G_0) + (2 * np.pi / beam.wavelength) * normG * sinth

            for hkl_indx in range(G_0.shape[1]):
                for time in laue.find_solutions_to_tangens_half_angle_equation( c_0s[hkl_indx], c_1s[hkl_indx], c_2s[hkl_indx], beam.rotator.alpha ):
                    if time is not None:
                        if utils.contained_by_intervals( time, proximity_intervals[ei] ):
                            element_vertices = rigid_body_motion( element_vertices_0, time )
                            scattering_region = beam.intersect( element_vertices )
                            if scattering_region is not None:
                                G = rigid_body_motion( G_0[:,hkl_indx], time )
                                scattered_wavevector = G_0[:,hkl_indx] + beam.wavevector
                                scatterer = Scatterer(  beam
                                                        scattering_region, 
                                                        scattered_wavevector, 
                                                        time, 
                                                        self.phases[ self.ephase[ei] ], 
                                                        hkl_indx  )
                                scatterers.append( scatterer )  
        detector.frames.append( scatterers )

        
