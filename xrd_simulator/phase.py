import numpy as np
from xrd_simulator.xfab import tools, structure
from xrd_simulator.utils import cif_open


class Phase(object):

    """Defines properties related to a crystal class.

    The barebone Phase object holds the space group name and unit cell of the crystal. From this it is possible
    to compute a set of Miller indices for a given wavelength that can give diffraction. In addition, it is
    possible to compute unit cell structure factors for the hkls which is usefull to model the scattered intensity.
    This will however require a CIF file.

    Args:
        unit_cell (:obj:`list` of :obj:`float`): Crystal unit cell representation of the form
            [a,b,c,alpha,beta,gamma], where alpha,beta and gamma are in units of degrees while
            a,b and c are in units of anstrom.
        sgname (:obj:`string`): Name of space group , e.g 'P3221' for quartz, SiO2, for instance
        path_to_cif_file (:obj:`string`): Path to CIF file. Defaults to None, in which case no structure
            factors are computed, i.e `structure_factors=None`.

    Attributes:
        unit_cell (:obj:`list` of :obj:`float`): Crystal unit cell representation of the form
            [a,b,c,alpha,beta,gamma], where alpha,beta and gamma are in units of degrees while
            a,b and c are in units of anstrom.
        sgname (:obj:`string`):  Name of space group , e.g 'P3221' for quartz, SiO2, for instance
        miller_indices (:obj:`numpy array`): Allowable integer Miller indices (h,k,l) of ``shape=(n,3)``.
        structure_factors (:obj:`numpy array`): Structure factors of allowable Miller indices (```miller_indices```)
            of ``shape=(n,2)``. `structure_factors[i,0]` gives the real structure factor of `hkl=miller_indices[i,:]`
            while `structure_factors[i,0]` gives the corresponding imaginary part of the structure factor.

    """

    def __init__(self, unit_cell, sgname, path_to_cif_file=None):
        self.unit_cell = unit_cell
        self.sgname = sgname
        self.miller_indices = None
        self.structure_factors = None
        self.path_to_cif_file = path_to_cif_file

    def setup_diffracting_planes(
            self,
            wavelength,
            min_bragg_angle,
            max_bragg_angle,
            verbose=True):
        """Generates all Miller indices (h,k,l) that will diffract given wavelength and Bragg angle bounds.

        If self.path_to_cif_file is not None, structure factors are computed in addition to the hkls.

        Args:
            wavelength (:obj:`float`): X-ray wavelength in units of anstrom.
            min_bragg_angle (:obj:`float`): Maximum Bragg angle, in radians, allowed to be taken during diffraction.
            max_bragg_angle (:obj:`float`): Minimum Bragg angle, in radians, allowed to be taken during diffraction.

        NOTE: This function will skip Miller indices that have a zero intensity due to the unit cell structure
        factor vanishing, i.e forbidden reflections, such as a 100 in an fcc for instance, will not be included.
        """
        sintlmin = min_bragg_angle / wavelength
        sintlmax = max_bragg_angle / wavelength
        self.miller_indices = tools.genhkl_all(
            self.unit_cell, sintlmin, sintlmax, sgname=self.sgname)
        if self.path_to_cif_file is not None:
            self._set_structure_factors(self.miller_indices, verbose=verbose)

    def _set_structure_factors(self, miller_indices, verbose):
        """Generate unit cell structure factors for all miller indices.
        """
        atom_factory = structure.build_atomlist()
        cifblk = cif_open(self.path_to_cif_file)
        atom_factory.CIFread(
            ciffile=None,
            cifblkname=None,
            cifblk=cifblk,
            verbose=verbose)
        atoms = atom_factory.atomlist.atom
        self.structure_factors = np.zeros((miller_indices.shape[0], 2))
        for i, hkl in enumerate(miller_indices):
            self.structure_factors[i, :] = structure.StructureFactor(
                hkl, self.unit_cell, self.sgname, atoms, disper=None)
