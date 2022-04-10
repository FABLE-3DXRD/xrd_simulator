from xrd_simulator.phase import Phase

# The phase can be specified via a crystalographic information file.
quartz = Phase(unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
               sgname='P3221',
               path_to_cif_file='quartz.cif'
               )