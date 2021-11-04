import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.utils import source
from xrd_simulator.scatterer import Scatterer
from xrd_simulator import utils
from xrd_simulator import laue
import copy

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
        
        # Assuming sample and lab frames to be aligned at instantiation.
        self.mesh_lab = copy.deepcopy(mesh)
        self.eU_lab = eU.copy()
        self.mesh_sample = copy.deepcopy(mesh)
        self.eU_sample = eU.copy()

        self.phases = phases
        self.ephase = ephase
        self.eB     = eB

    def diffract(self, beam, detector, rigid_body_motion, min_bragg_angle=0, max_bragg_angle=None):
        """Construct the scattering regions in the wave vector range [k1,k2] given a beam profile.

        The beam interacts with the polycrystal producing scattering captured by the detector.

        Args:
            beam (:obj:`xrd_simulator.beam.Beam`): Object representing a monochromatic beam of X-rays.
            detector (:obj:`xrd_simulator.detector.Detector`): Object representing a flat rectangular detector.
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a funciton of time on the domain time=[0,1].
            min_bragg_angle (:obj:`float`): Time between [0,1] at which to call the rigid body motion. Defaults to 0.
            max_bragg_angle (:obj:`float`): Time between [0,1] at which to call the rigid body motion. Defaults to a 
                guess value approximated by wrapping the detector corners in a cone with apex in the sample.

        """

        min_bragg_angle, max_bragg_angle = self._get_bragg_angle_bounds(detector, beam, min_bragg_angle, max_bragg_angle)

        for phase in self.phases:
            phase.setup_diffracting_planes(beam.wavelength, min_bragg_angle, max_bragg_angle) 
        
        c_0_factor   = -beam.wave_vector.dot( rigid_body_motion.rotator.K2 )
        c_1_factor   =  beam.wave_vector.dot( rigid_body_motion.rotator.K  )
        c_2_factor   =  beam.wave_vector.dot( rigid_body_motion.rotator.I + rigid_body_motion.rotator.K2 )

        scatterers = []

        proximity_intervals = beam.get_proximity_intervals(self.mesh_lab.espherecentroids, self.mesh_lab.eradius, rigid_body_motion) 

        for ei in range( self.mesh_lab.number_of_elements ):

            # skipp elements not illuminated
            if proximity_intervals[ei][0] is None: 
                continue
            

            print("Computing for element {} of total elements {}".format(ei,self.mesh_lab.number_of_elements))
            element_vertices_0 = self.mesh_lab.coord[self.mesh_lab.enod[ei],:] 
            G_0 = laue.get_G(self.eU_lab[ei], self.eB[ei], self.phases[ self.ephase[ei] ].miller_indices.T )

            c_0s  = c_0_factor.dot(G_0)
            c_1s  = c_1_factor.dot(G_0)
            c_2s  = c_2_factor.dot(G_0) + np.sum((G_0*G_0),axis=0)/2.

            for hkl_indx in range(G_0.shape[1]):
                for time in laue.find_solutions_to_tangens_half_angle_equation( c_0s[hkl_indx], c_1s[hkl_indx], c_2s[hkl_indx], rigid_body_motion.rotation_angle ):
                    if time is not None:
                        if utils.contained_by_intervals( time, proximity_intervals[ei] ):
                            element_vertices = rigid_body_motion( element_vertices_0.T, time ).T

                            # mark elements that neaver leave the beam before computing intersection
                            scattering_region = beam.intersect( element_vertices )

                            if scattering_region is not None:
                                G = rigid_body_motion.rotate( G_0[:,hkl_indx], time )
                                scattered_wave_vector = G + beam.wave_vector                                
                                scatterer = Scatterer(  scattering_region, 
                                                        scattered_wave_vector,
                                                        beam.wave_vector,
                                                        beam.wavelength,
                                                        beam.polarization_vector,
                                                        rigid_body_motion.rotation_axis,
                                                        time, 
                                                        self.phases[ self.ephase[ei] ], 
                                                        hkl_indx )
                                scatterers.append( scatterer )
        detector.frames.append( scatterers )

    def _get_bragg_angle_bounds(self, detector, beam, min_bragg_angle, max_bragg_angle):

        if max_bragg_angle is None:
            # TODO: make a smarter selection of source point for get_wrapping_cone()
            max_bragg_angle = detector.get_wrapping_cone( beam.wave_vector, self.mesh_lab.centroid )

        assert min_bragg_angle>=0, "min_bragg_angle must be greater or equal than zero"
        assert max_bragg_angle>min_bragg_angle, "max_bragg_angle must be greater than min_bragg_angle"

        if max_bragg_angle > 25*np.pi/180: 
            angle = str(np.round(np.degrees(max_bragg_angle),1))
            print( "WARNING: Maximum Bragg-angle is large ("+angle+" dgrs), computations may be slow due to abundant scattering." )
        
        return min_bragg_angle, max_bragg_angle

    def transform(self, rigid_body_motion, time):
        """Transform the polycrystal by performing a rigid body motion (translation + rotation)

        This function will update the polycrystal mesh (update in lab frame) with any dependent quanteties, 
        such as face normals etc. Likewise, it will update the per element crystallite orientation 
        matrices (U) such that the updated matrix will

        Args:
            rigid_body_motion (:obj:`xrd_simulator.motion.RigidBodyMotion`): Rigid body motion object describing the
                polycrystal transformation as a funciton of time on the domain time=[0,1].
            time (:obj:`float`): Time between [0,1] at which to call the rigid body motion.

        
        """
        new_nodal_coordinates = rigid_body_motion( self.mesh_lab.coord.T, time=time )
        self.mesh_lab.update( new_nodal_coordinates )
        for ei in range(self.mesh_lab.coord.shape[0]):
            self.eU_lab[ei] = rigid_body_motion.rotate( self.eU_lab[ei], time=time )
