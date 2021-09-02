"""A scatterer object defines the convex scattering region of an element together with
its scattering vector kprime, intensity scaling and scan sequence position. It is meant to be 
eaten by a detector and rendered into a pattern. The scatterd ray object is not a simplified 
object but any approximations are meant to be made during detector rendering."""

import numpy as np 
import matplotlib.pyplot as plt

class Scatterer(object):

    def __init__(self, halfspaceintersection, kprime ):
        self.halfspaceintersection = halfspaceintersection
        self.kprime = kprime
    
    def get_centroid(self):
        """Get centroid scattering region
        """
        pass

    def get_volume(self):
        """Get centroid scattering region
        """
        pass