'''Library of commonly used crystallographic phases for xrd_simulator'''
from xrd_simulator.phase import Phase
import os

dir_this_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),'artifacts','cif_files')

quartz = Phase(unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
               sgname='P3221',  # (Quartz)
            #    path_to_cif_file='http://www.crystallography.net/cod/5000035.cif'
               path_to_cif_file=os.path.join(dir_this_file,'Quartz.cif')
               )

a_Ferrite = Phase(unit_cell=[2.8680182, 2.8680182, 2.8680182, 90., 90., 90.],
               sgname='Im-3m',  # (Iron)
               path_to_cif_file=os.path.join(dir_this_file,'Fe_bcc.cif')
               )

Austenite = Phase(unit_cell=[3.581, 3.581, 3.581, 90., 90., 90.],
               sgname='Fm-3m',  # (Steel)
               path_to_cif_file=os.path.join(dir_this_file,'Fe_fcc.cif')
               )

Oleg = Phase(unit_cell=[3.581, 3.581, 3.581, 90., 90., 90.],
               sgname='Fm-3m',  # (Steel)
               path_to_cif_file=os.path.join(dir_this_file,'Fe_fcc_oleg.cif')
               )

hcp = Phase(unit_cell=[2.534, 2.534, 3.941, 90., 90., 120.],
               sgname='P63/mmc',  # (Steel)
               path_to_cif_file=os.path.join(dir_this_file,'Fe_hcp_modified.cif')
               )

# Martensite = Phase(unit_cell=[4.009, 4.009, 36.672, 90., 90., 90.],
#                sgname='Pmna',  # (Steel)
#                path_to_cif_file='./cif_files/1532971.cif'
#                )

Cementite = Phase(unit_cell=[4.5144, 5.0787, 6.7297, 90., 90., 90.],
               sgname='Pnma',  # (Fe3C)
               path_to_cif_file=os.path.join(dir_this_file,'Cementite.cif')
               )
