import numpy as np 
import matplotlib.pyplot as plt
from xfab import tools, structure
from xrd_simulator.utils import _HiddenPrints

class Phase(object):

    """Defines properties related to a crystal class.

    The barebone Phase object holds the space group name and unit cell of the crystal. From this it is possible
    to compute a set of Miller indices for a given wavelength that can give diffraction. In addition, it is 
    possible to compute unit cell structure factors for the hkls which is usefull to model the scattered intensity.
    This will however require a CIF file.

    Args:
        unit_cell (:obj:`list`): Crystal unit cell representation of the form [a,b,c,alpha,beta,gamma], where 
            alpha,beta and gamma are in units of degrees while a,b and c are in units of anstrom.
        sgname (:obj:`string`): Name of space group , e.g 'P3221' for quartz, SiO2, for instance

    Attributes:
        unit_cell (:obj:`list`): Crystal unit cell representation of the form [a,b,c,alpha,beta,gamma], where 
            alpha,beta and gamma are in units of degrees while a,b and c are in units of anstrom.
        sgname (:obj:`string`):  Name of space group , e.g 'P3221' for quartz, SiO2, for instance
        miller_indices (:obj:`numpy array`): Allowable integer Miller indices (h,k,l) of ```shape=(n,3)```. 
        structure_factors (:obj:`numpy array`): Structure factors of allowable Miller indices (```miller_indices```)
            of ```shape=(n,2)```. `structure_factors[i,0]` gives the real structure factor of `hkl=miller_indices[i,:]` 
            while `structure_factors[i,0]` gives the corresponding imaginary part of the structure factor.

    """

    def __init__( self, unit_cell, sgname ):
        self.unit_cell = unit_cell
        self.sgname = sgname
        miller_indices    = None
        structure_factors = None

    def set_miller_indices(self, wavelength, min_bragg_angle, max_bragg_angle):
        """Generate all Miller indices (h,k,l) that will difract given wavelength and Bragg angle bounds.

        Args:
            wavelength (:obj:`float`): X-ray wavelenght in units of anstrom.
            min_bragg_angle (:obj:`float`): Maximum Bragg angle, in radians, allowed to be taken during diffraction.
            max_bragg_angle (:obj:`float`): Minimum Bragg angle, in radians, allowed to be taken during diffraction.

        NOTE: This function will skip Miller indices that have a zero intensity due to the unit cell structure 
        factor vanishing, i.e forbidden reflections, such as a 100 in an fcc for instance, will not be included.
        """
        sintlmin = min_bragg_angle / wavelength
        sintlmax = max_bragg_angle / wavelength
        with _HiddenPrints(): #TODO: perhaps suggest xfab not to print in the first place and make a pull request.
            self.miller_indices = tools.genhkl_all(self.unit_cell, sintlmin, sintlmax, sgname=self.sgname)

    def set_structure_factors(self, path_to_cif_file):
        """Generate unit cell structure factors for all miller indices.

        Args:
            path_to_cif_file (:obj:`string`): Path to Crystallographic Information File (CIF).

        """

        if self.miller_indices is None:
            raise ValueError("The Phase object have no Miller indices, use set_miller_indices() to generate hkls.")

        atom_factory = structure.build_atomlist()
        atom_factory.CIFread(ciffile = path_to_cif_file, cifblkname = None, cifblk = None)
        atoms = atom_factory.atomlist.atom

        self.structure_factors = np.zeros((self.miller_indices.shape[0], 2))
        for i,hkl in enumerate(self.miller_indices):
            self.structure_factors[i,:] = structure.StructureFactor(hkl, self.unit_cell, self.sgname, atoms, disper=None)
