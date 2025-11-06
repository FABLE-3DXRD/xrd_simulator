'''Script to create beam components to be used within the xrd_simulator package. 
They are stored by default in the /artifacts/beams/ folder within the current directory'''

import os
import numpy as np
from xrd_simulator.beam import Beam

# Beam parameters
width = 2000 # um
height = 2000 # um
length = 2000 # um
energy = 23 # keV

dir_this_file=os.path.dirname(os.path.abspath(__file__))
destination = os.path.join(dir_this_file,'artifacts','beams')
vector = [1, 0, 0]
save_file_path = os.path.join(destination,f'{energy}keV_{width}x{height}x{length}')

# Conversion to wavelength
wavelength = 1.2398E-9/energy

beam_vertices = np.array([
    [-length*0.5, -width*0.5, -height*0.5],
    [-length*0.5, width*0.5, -height*0.5],
    [-length*0.5, width*0.5, height*0.5],
    [-length*0.5, -width*0.5, height*0.5],
    [length*0.5, -width*0.5, -height*0.5],
    [length*0.5, width*0.5, -height*0.5],
    [length*0.5, width*0.5, height*0.5],
    [length*0.5, -width*0.5, height*0.5]])

beam = Beam(
    beam_vertices,
    xray_propagation_direction=np.array(vector),
    wavelength=wavelength,
    polarization_vector=np.array([0., 1., 0.]))

if not os.path.exists(destination):
    os.makedirs(destination)
    

beam.save(save_file_path)
print(f'File saved in {save_file_path}')