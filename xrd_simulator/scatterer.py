"""A scatterer object defines the convex scattering region of an element together with
its scattering vector kprime, intensity scaling and scan sequence position. It is meant to be 
eaten by a detector and rendered into a pattern. The scatterd ray object is not a simplified 
object but any approximations are meant to be made during detector rendering."""

import numpy as np 
import matplotlib.pyplot as plt

class Scatterer(object):

    def __init__(self, convex_hull, kprime, s ):
        self.convex_hull = convex_hull
        self.kprime = kprime
        self.s = s
    
    def get_centroid(self):
        """Get centroid scattering region
        """
        return np.mean( convex_hull.vertices, axis=0 )

    def get_volume(self):
        """Get centroid scattering region
        """
        return convex_hull.volume