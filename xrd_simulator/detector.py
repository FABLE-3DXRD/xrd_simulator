"""The detector class takes a set of scattere-rays objects and produces a 
discrete pixelated framestack."""

import numpy as np 
import matplotlib.pyplot as plt

class Detector(object):

    def __init__(self, pixel_size, geometry_matrix ):
        self.pixel_size      = pixel_size
        self.geometry_matrix = geometry_matrix
        self.frames = []
    
    def render(self, scatterers):
        """Take a list of scatterers render and add to the frames list.
        """
        pass

    
