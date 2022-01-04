"""
xfab.structure for reading crystal structure files (cif,pdb)
and calculation of form factors and structure factors etc.
"""

from __future__ import absolute_import
import numpy as n
import logging

from xrd_simulator.xfab import tools
from xrd_simulator.xfab import sg
from xrd_simulator.xfab import atomlib


def StructureFactor(hkl, ucell, sgname, atoms, disper=None):
    """
    Calculation of the structure factor of reflection hkl

    [Freal Fimg] = StructureFactor(hkl,unit_cell,sg,atoms)

    INPUT : hkl =       [h, k, l]
            unit_cell = [a, b, c, alpha, beta, gamma]
            sgname:     space group name (e.g. 'P 21/c')
            atoms:      structural parameters (as an object)
    OUTPUT: The real and imaginary parts of the the structure factor

    Henning Osholm Sorensen, June 23, 2006.
    Translated to python code April 8, 2008
    """
    mysg = sg.sg(sgname=sgname)
    stl = tools.sintl(ucell, hkl)
    noatoms = len(atoms)

    Freal = 0.0
    Fimg = 0.0

    for i in range(noatoms):
        # Check whether isotrop or anisotropic displacements
        if atoms[i].adp_type == 'Uiso':
            U = atoms[i].adp
            expij = n.exp(-8 * n.pi**2 * U * stl**2)
        elif atoms[i].adp_type == 'Uani':
            # transform Uij to betaij
            betaij = Uij2betaij(atoms[i].adp, ucell)
        else:
            expij = 1
            atoms[i].adp = 'Uiso'
            #logging.error("wrong no of elements in atomlist")

        # Atomic form factors
        f = FormFactor(atoms[i].atomtype, stl)
        if disper is None or disper[atoms[i].atomtype] is None:
            fp = 0.0
            fpp = 0.0
        else:
            fp = disper[atoms[i].atomtype][0]
            fpp = disper[atoms[i].atomtype][1]

        for j in range(mysg.nsymop):
            # atomic displacement factor
            if atoms[i].adp_type == 'Uani':
                betaijrot = n.dot(mysg.rot[j], n.dot(betaij, mysg.rot[j]))
                expij = n.exp(-n.dot(hkl, n.dot(betaijrot, hkl)))

            # exponent for phase factor
            r = n.dot(mysg.rot[j], atoms[i].pos) + mysg.trans[j]
            exponent = 2 * n.pi * n.dot(hkl, r)

            # forming the real and imaginary parts of F
            s = n.sin(exponent)
            c = n.cos(exponent)
            site_pop = atoms[i].occ * atoms[i].symmulti / mysg.nsymop
            Freal = Freal + expij * (c * (f + fp) - s * fpp) * site_pop
            Fimg = Fimg + expij * (s * (f + fp) + c * fpp) * site_pop

    return [Freal, Fimg]


def Uij2betaij(adp, ucell):
    """
    Uij2betaij transform the ADP U-matrix into the beta form

    betaij = Uij2betaij(adp,unit_cell)

    INPUT:  adp: anisotropic displacement parameter U matrix
            Uijs are given in this order: [U11, U22, U33, U23, U13, U12]
            unit_cell = [a, b, c, alpha, beta, gamma]

    OUTPUT: betaij: beta displacement matrix

    Henning Osholm Sorensen, Risoe National Laboratory, June 23, 2006.
    Translated to python code March 29, 2008
    """
    U = n.array([[adp[0], adp[5], adp[4]],
                 [adp[5], adp[1], adp[3]],
                 [adp[4], adp[3], adp[2]]])

    betaij = n.zeros((3, 3))
    cellstar = tools.cell_invert(ucell)

    for i in range(3):
        for j in range(3):
            betaij[i, j] = 2 * n.pi**2 * cellstar[i] * cellstar[j] * U[i, j]

    return betaij


def FormFactor(atomtype, stl):
    """
     Calculation of the atomic form factor at a specified sin(theta)/lambda
     using the analytic fit to the  form factors from
     Int. Tab. Cryst Sect. C 6.1.1.4

     INPUT:   atomtype: Atom type (string) e.g. 'C'
              stl: form factor calculated sin(theta)/lambda = stl
     OUTPUT:  atomic form factor (no dispersion)


     Henning Osholm Sorensen, Risoe National Laboratory, April 9, 2008.

    """

    data = atomlib.formfactor[atomtype]

    # Calc form factor
    formfac = 0

    for i in range(4):
        formfac = formfac + data[i] * n.exp(-data[i + 4] * stl * stl)
    formfac = formfac + data[8]

    return formfac


def int_intensity(F2, L, P, I0, wavelength, cell_vol, cryst_vol):
    """
    Calculate the reflection intensities scaling factor

    INPUT:
    F2        : the structure factor squared
    L         : Lorentz factor
    P         : Polarisation factor
    I0        : Incoming beam flux
    wavelength: in Angstroem
    cell_vol  : Volume of unit cell in AA^3
    cryst_vol : Volume of crystal in mm^3

    OUTPUT:
    int_intensity: integrated intensity

    """

    emass = 9.1093826e-31
    echarge = 1.60217653e-19
    pi4eps0 = 1.11265e-10
    c = 299792458.0
    k1 = (echarge**2 / (pi4eps0 * emass * c**2) * 1000)**2  # Unit is mm
    # the factor 1e21 used below is to go from mm^3 to AA^3
    k2 = wavelength**3 * cryst_vol * 1e21 / cell_vol**2
    return k1 * k2 * I0 * L * P * F2


def multiplicity(position, sgname=None, sgno=None, cell_choice='standard'):
    """
    Calculates the multiplicity of a fractional position in the unit cell.
    If called by sgno, cell_choice is necessary for eg rhombohedral space groups.

    """

    if sgname is not None:
        mysg = sg.sg(sgname=sgname, cell_choice=cell_choice)
    elif sgno is not None:
        mysg = sg.sg(sgno=sgno, cell_choice=cell_choice)
    else:
        raise ValueError('No space group information provided')

    lp = n.zeros((mysg.nsymop, 3))

    for i in range(mysg.nsymop):
        lp[i, :] = n.dot(position, mysg.rot[i]) + mysg.trans[i]

    lpu = n.array([lp[0, :]])
    multi = 1

    for i in range(1, mysg.nsymop):
        for j in range(multi):
            t = lp[i] - lpu[j]
            if n.sum(n.mod(t, 1)) < 0.00001:
                break
            else:
                if j == multi - 1:
                    lpu = n.concatenate((lpu, [lp[i, :]]))
                    multi += 1
    return multi


class atom_entry:
    def __init__(self, label=None, atomtype=None, pos=None,
                 adp_type=None, adp=None, occ=None, symmulti=None):
        self.label = label
        self.atomtype = atomtype
        self.pos = pos
        self.adp_type = adp_type
        self.adp = adp
        self.occ = occ
        self.symmulti = symmulti


class atomlist:
    def __init__(self, sgname=None, sgno=None, cell=None):
        self.sgname = sgname
        self.sgno = sgno
        self.cell = cell
        self.dispersion = {}
        self.atom = []
    def add_atom(self, label=None, atomtype=None, pos=None,
                 adp_type=None, adp=None, occ=None, symmulti=None):
        self.atom.append(atom_entry(label=label, atomtype=atomtype,
                                    pos=pos, adp_type=adp_type,
                                    adp=adp, occ=occ, symmulti=symmulti))


class build_atomlist:
    def __init__(self):
        self.atomlist = atomlist()

    def CIFopen(self, ciffile=None, cifblkname=None):
        from CifFile import ReadCif  # part of the PycifRW module
        try:
            # the following is a trick to avoid that urllib.URLopen.open
            # used by ReadCif misinterprets the url when a drive:\ is
            # present (Win)
            ciffile = ciffile.replace(':', '|')
            cf = ReadCif(ciffile)
        except BaseException:
            logging.error('File %s could not be accessed' % ciffile)

        if cifblkname is None:
            # Try to guess blockname
            blocks = list(cf.keys())
            if len(blocks) > 1:
                if len(blocks) == 2 and 'global' in blocks:
                    cifblkname = blocks[abs(blocks.index('global') - 1)]
                else:
                    logging.error('More than one possible data set:')
                    logging.error(
                        'The following data block names are in the file:')
                    for block in blocks:
                        logging.error(block)
                    raise Exception
            else:
                # Only one available
                cifblkname = blocks[0]
        # Extract block
        try:
            self.cifblk = cf[cifblkname]
        except BaseException:
            logging.error(
                'Block - %s - not found in %s' %
                (cifblkname, ciffile))
            raise IOError
        return self.cifblk

    def PDBread(self, pdbfile=None):
        """
        function to read pdb file (www.pdb.org) and make
        atomlist structure
        """
        from re import sub
        try:
            text = open(pdbfile, 'r').readlines()
        except BaseException:
            logging.error('File %s could not be accessed' % pdbfile)

        for i in range(len(text)):
            if text[i].find('CRYST1') == 0:
                a = float(text[i][6:15])
                b = float(text[i][15:24])
                c = float(text[i][24:33])
                alp = float(text[i][33:40])
                bet = float(text[i][40:47])
                gam = float(text[i][47:54])
                sg = text[i][55:66]

        self.atomlist.cell = [a, b, c, alp, bet, gam]

        # Make space group name
        sgtmp = sg.split()
        sg = ''
        for i in range(len(sgtmp)):
            if sgtmp[i] != '1':
                sg = sg + sgtmp[i].lower()
        self.atomlist.sgname = sg

        # Build SCALE matrix for transformation of
        # orthonormal atomic coordinates to fractional
        scalemat = n.zeros((3, 4))
        for i in range(len(text)):
            if text[i].find('SCALE') == 0:
                # FOUND SCALE LINE
                scale = text[i].split()
                scaleline = int(scale[0][-1]) - 1
                for j in range(1, len(scale)):
                    scalemat[scaleline, j - 1] = float(scale[j])

        no = 0
        for i in range(len(text)):
            if text[i].find('ATOM') == 0 or text[i].find('HETATM') == 0:
                no = no + 1
                label = sub("\\s+", "", text[i][12:16])
                atomtype = sub("\\s+", "", text[i][76:78]).upper()
                x = float(text[i][30:38])
                y = float(text[i][38:46])
                z = float(text[i][46:54])
                # transform orthonormal coordinates to fractional
                pos = n.dot(scalemat, [x, y, z, 1])
                adp = float(text[i][60:66]) / (8 * n.pi**2)  # B to U
                adp_type = 'Uiso'
                occ = float(text[i][54:60])
                multi = multiplicity(pos, self.atomlist.sgname)
                self.atomlist.add_atom(label=label,
                                       atomtype=atomtype,
                                       pos=pos,
                                       adp_type=adp_type,
                                       adp=adp,
                                       occ=occ,
                                       symmulti=multi)

                self.atomlist.dispersion[atomtype] = None

    def CIFread(
            self,
            ciffile=None,
            cifblkname=None,
            cifblk=None,
            verbose=True):
        from re import sub
        if ciffile is not None:
            try:
                cifblk = self.CIFopen(ciffile=ciffile, cifblkname=cifblkname)
            except BaseException:
                raise
        elif cifblk is None:
            cifblk = self.cifblk

        self.atomlist.cell = [self.remove_esd(cifblk['_cell_length_a']),
                              self.remove_esd(cifblk['_cell_length_b']),
                              self.remove_esd(cifblk['_cell_length_c']),
                              self.remove_esd(cifblk['_cell_angle_alpha']),
                              self.remove_esd(cifblk['_cell_angle_beta']),
                              self.remove_esd(cifblk['_cell_angle_gamma'])]

        # self.atomlist.sgname = upper(sub("\s+","",
        #                       cifblk['_symmetry_space_group_name_H-M']))
        self.atomlist.sgname = sub("\\s+", "",
                                   cifblk['_symmetry_space_group_name_H-M'])

        # Dispersion factors
        if '_atom_type_symbol' in list(cifblk.keys()):
            for i in range(len(cifblk['_atom_type_symbol'])):
                try:
                    self.atomlist.dispersion[cifblk['_atom_type_symbol'][i].upper()] =\
                        [self.remove_esd(cifblk['_atom_type_scat_dispersion_real'][i]),
                         self.remove_esd(cifblk['_atom_type_scat_dispersion_imag'][i])]
                except BaseException:
                    self.atomlist.dispersion[cifblk['_atom_type_symbol'][i].upper(
                    )] = None
                    if verbose:
                        logging.warning(
                            'No dispersion factors for %s in cif file - set to zero' %
                            cifblk['_atom_type_symbol'][i])
        else:
            if verbose:
                logging.warning('No _atom_type_symbol found in CIF')
            for i in range(len(cifblk['_atom_site_type_symbol'])):
                self.atomlist.dispersion[cifblk['_atom_site_type_symbol'][i].upper(
                )] = None
                if verbose:
                    logging.warning(
                        'No dispersion factors for %s in cif file - set to zero' %
                        cifblk['_atom_site_type_symbol'][i])

        for i in range(len(cifblk['_atom_site_type_symbol'])):
            label = cifblk['_atom_site_label'][i]
            #atomno = atomtype[upper(cifblk['_atom_site_type_symbol'][i])]
            atomtype = cifblk['_atom_site_type_symbol'][i].upper()
            x = self.remove_esd(cifblk['_atom_site_fract_x'][i])
            y = self.remove_esd(cifblk['_atom_site_fract_y'][i])
            z = self.remove_esd(cifblk['_atom_site_fract_z'][i])
            try:
                adp_type = cifblk['_atom_site_adp_type'][i]
            except BaseException:
                adp_type = None
            try:
                occ = self.remove_esd(cifblk['_atom_site_occupancy'][i])
            except BaseException:
                occ = 1.0

            if '_atom_site_symmetry_multiplicity' in cifblk:
                multi = self.remove_esd(
                    cifblk['_atom_site_symmetry_multiplicity'][i])
            # In old SHELXL versions this code was written
            # as '_atom_site_symetry_multiplicity'
            elif '_atom_site_symetry_multiplicity' in cifblk:
                multi = self.remove_esd(
                    cifblk['_atom_site_symetry_multiplicity'][i])
            else:
                multi = multiplicity([x, y, z], self.atomlist.sgname)

            # Test for B or U

            if adp_type is None:
                adp = 0.0
            elif adp_type == 'Biso':
                adp = self.remove_esd(
                    cifblk['_atom_site_B_iso_or_equiv'][i]) / (8 * n.pi**2)
                adp_type = 'Uiso'
            elif adp_type == 'Bani':
                anisonumber = cifblk['_atom_site_aniso_label'].index(label)
                adp = [self.remove_esd(cifblk['_atom_site_aniso_B_11'][anisonumber]) / (8 * n.pi**2),
                       self.remove_esd(cifblk['_atom_site_aniso_B_22'][anisonumber]) / (8 * n.pi**2),
                       self.remove_esd(cifblk['_atom_site_aniso_B_33'][anisonumber]) / (8 * n.pi**2),
                       self.remove_esd(cifblk['_atom_site_aniso_B_23'][anisonumber]) / (8 * n.pi**2),
                       self.remove_esd(cifblk['_atom_site_aniso_B_13'][anisonumber]) / (8 * n.pi**2),
                       self.remove_esd(cifblk['_atom_site_aniso_B_12'][anisonumber]) / (8 * n.pi**2)]
                adp_type = 'Uani'
            elif adp_type == 'Uiso':
                adp = self.remove_esd(cifblk['_atom_site_U_iso_or_equiv'][i])
            elif adp_type == 'Uani':
                anisonumber = cifblk['_atom_site_aniso_label'].index(label)
                adp = [self.remove_esd(cifblk['_atom_site_aniso_U_11'][anisonumber]),
                       self.remove_esd(cifblk['_atom_site_aniso_U_22'][anisonumber]),
                       self.remove_esd(cifblk['_atom_site_aniso_U_33'][anisonumber]),
                       self.remove_esd(cifblk['_atom_site_aniso_U_23'][anisonumber]),
                       self.remove_esd(cifblk['_atom_site_aniso_U_13'][anisonumber]),
                       self.remove_esd(cifblk['_atom_site_aniso_U_12'][anisonumber])]
            self.atomlist.add_atom(label=label, atomtype=atomtype,
                                   pos=[x, y, z], adp_type=adp_type,
                                   adp=adp, occ=occ, symmulti=multi)

    def remove_esd(self, a):
        """
        This function will remove the esd part of the entry,
        e.g. '1.234(56)' to '1.234'.
        """

        if a.find('(') == -1:
            value = float(a)
        else:
            value = float(a[:a.find('(')])
        return value
