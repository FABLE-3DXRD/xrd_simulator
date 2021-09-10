import numpy as np 
import matplotlib.pyplot as plt
from xfab import tools
from xrd_simulator.utils import _HiddenPrints

class Phase(object):

    """Defines properties related to a crystal class, including a genration of Miller indices.

    Args:
        unit_cell (:obj:`list`): Crystal unit cell representation of the form [a,b,c,alpha,beta,gamma], where 
            alpha,beta and gamma are in units of degrees while a,b and c are in units of anstrom.
        sgname (:obj:`string`): Name of space group , e.g 'P3221' for quartz, SiO2, for instance

    Attributes:
        unit_cell (:obj:`list`): Crystal unit cell representation of the form [a,b,c,alpha,beta,gamma], where 
            alpha,beta and gamma are in units of degrees while a,b and c are in units of anstrom.
        sgname (:obj:`string`):  Name of space group , e.g 'P3221' for quartz, SiO2, for instance

    """ 

    def __init__(self, unit_cell, sgname ):
        self.unit_cell = unit_cell
        self.sgname = sgname

    def generate_miller_indices(self, wavelength, min_bragg_angle, max_bragg_angle):
        """Generate all Miller indices (h,k,l) that will difract given wavelength and Bragg angle bounds.

        Args:
            wavelength (:obj:`float`): X-ray wavelenght in units of anstrom.
            min_bragg_angle (:obj:`float`): Maximum Bragg angle, in radians, allowed to be taken during diffraction.
            max_bragg_angle (:obj:`float`): Minimum Bragg angle, in radians, allowed to be taken during diffraction.

        Returns:
            (:obj:`numpy array`): Integer Miller indices (h,k,l) of ```shape=(N,3)```. Each row is a diffracting set of planes.

        """
        sintlmin = min_bragg_angle / wavelength
        sintlmax = max_bragg_angle / wavelength
        with _HiddenPrints(): #TODO: perhaps suggest xfab not to print in the first place and make a pull request.
            miller_indices = tools.genhkl_all(self.unit_cell, sintlmin, sintlmax, sgname=self.sgname)
        return miller_indices

    def get_structure_factors(self, hkl):
        """Generate Structure factors for a list of h,k,l indices.

        Args:
            hkl (:obj:`list` of :obj:`numpy array`): 

        Returns:
            
        """
        # TODO: use xfab.structure to compute these, requires CIF files and atomlists.
        raise NotImplementedError()
