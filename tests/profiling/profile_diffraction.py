import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xfab import tools
import cProfile
import pstats

""" Profile the 
"""

np.random.seed(10)

totrot = np.pi/180

pixel_size = 75.
detector_size = pixel_size*1024
detector_distance = 142938.28756189224
geometry_matrix_0 = np.array([
    [detector_distance,   -detector_size/2.,  -detector_size/2.],
    [detector_distance,    detector_size/2.,  -detector_size/2.],
    [detector_distance,   -detector_size/2.,   detector_size/2.]]).T
def geometry_descriptor(s):
    sin = np.sin( -s*totrot/2. )
    cos = np.cos( -s*totrot/2. )
    Rz = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])
    return Rz.dot( geometry_matrix_0 )
detector = Detector( pixel_size, geometry_descriptor )

# l = detector_size/10.
# coord = np.array([ [0,0,0],
#                    [0,l,0],
#                    [l,0,0],
#                    [0,0,l] ])    
# enod = np.array([[0,1,2,3]])
# mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)

mesh = TetraMesh.generate_mesh_from_levelset(
    level_set = lambda x: np.dot( x, x ) - detector_size/10.,
    bounding_radius = 1.1*detector_size/10., 
    cell_size = 0.0065*detector_size/10. )


#TODO: change this path 
# mesh.to_xdmf("/home/axel/workspace/xrd_simulator/tests/visual_tests/quartz")

unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
sgname = 'P3221' # Quartz
phases = [Phase(unit_cell, sgname)]
B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )
euler_angles = np.random.rand(mesh.number_of_elements, 3) * 2 * np.pi
eU = np.array( [tools.euler_to_u(ea[0], ea[1], ea[2]) for ea in euler_angles] )
ephase = np.zeros((mesh.number_of_elements,)).astype(int)
polycrystal = Polycrystal(mesh, ephase, eU, eB, phases)

w = detector_size/40. # partial illumination with pencil beam
beam_vertices = np.array([
    [-detector_distance, -w, -w ],
    [-detector_distance,  w, -w ],
    [-detector_distance,  w,  w ],
    [-detector_distance, -w,  w ],
    [ detector_distance, -w, -w  ],
    [ detector_distance,  w, -w  ],
    [ detector_distance,  w,  w  ],
    [ detector_distance, -w,  w  ]])
wavelength = 0.285227

sin = np.sin( -totrot/2. )
cos = np.cos( -totrot/2. )
Rz = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])

k1 = np.array([1,0,0]) * 2 * np.pi / wavelength
k2 = Rz.dot(k1)
beam = Beam(beam_vertices, wavelength, k1, k2, translation=np.array([0., 0., 0.]))

pr = cProfile.Profile()
pr.enable()
polycrystal.diffract( beam, detector )
pr.disable()
pr.dump_stats('profile_dump')
ps = pstats.Stats('profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(20)

pixim = detector.render(frame_number=0)

import matplotlib.pyplot as plt
from scipy.signal import convolve
kernel = np.ones((5,5))*(1/10.)
kernel[1,1] = (2/10.)
pixim = convolve(pixim, kernel, mode='full', method='auto')
plt.imshow(pixim, cmap='gray')
plt.title("Hits: "+str(len(detector.frames[0]) ))
plt.show()
