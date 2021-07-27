"""A Polycrystal object links a mesh to a phase-list and hold a ubi matrix to each element
and can produce corresponding hkl indices and intensity factors for these crystals""" 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class Polycrystal(object):

    def __init__(self, mesh, ephase, phases ):
        self.mesh   = mesh
        self.ephase = ephase
        self.phases = phases

    def _get_crystal_beam_intersections(self, beam, elements):
        """Compute convex intersection hulls of beam and selected mesh elements."""
        pass

    def _get_candidate_elements(self, beam):
        """Get all elements that could diffract for a given illumination setting."""
        pass

    def get_scattered_rays(self, beam):
        """Construct ScatteredRay objects for each scattering occurence and return."""
        pass
