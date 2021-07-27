"""The beam class defines an X-ray as a 3d spatial (convex) object by 
taking a set of points and intepreting them as the convex hull of the beam
cross section. The beam is then extruded in a given direction towards infinity.""" 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class Beam(object):

    def __init__(self, vertices, wavelength ):
        self.convex_beam_hull = ConvexHull(vertices)
        self.wavelength = wavelength