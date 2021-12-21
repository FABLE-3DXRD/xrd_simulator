import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xfab import tools
from scipy.signal import convolve

pixel_size = 75.
detector_size = pixel_size*1024
detector_distance = 142938.28756189224
d0 = np.array([detector_distance,   -detector_size/2.,  -detector_size/2.])
d1 = np.array([detector_distance,    detector_size/2.,  -detector_size/2.])
d2 = np.array([detector_distance,   -detector_size/2.,   detector_size/2.])

detector = Detector( pixel_size, pixel_size, d0, d1, d2 )

# mesh = TetraMesh.generate_mesh_from_levelset(
#     level_set = lambda x: pixel_size*x[0]*x[0] + pixel_size*x[1]*x[1] + x[2]*x[2] - detector_size/10.,
#     bounding_radius = 1.1*detector_size/10., 
#     max_cell_circumradius = 0.001*detector_size/10. )

coord, enod = [],[]
k=0
dx = 0.001*detector_size/10.
c = np.array([0,0,0])
for _ in range(500):
    coord.append( [c[0],   c[1],   c[2]] ) 
    coord.append( [c[0]+dx,   c[1],   c[2]] ) 
    coord.append( [c[0],   c[1]+dx,   c[2]] ) 
    coord.append( [c[0],   c[1],   c[2]+dx] )
    enod.append( [k,k+1,k+2,k+3] )
    k+=3
coord = np.array(coord)
enod = np.array(enod)
mesh = TetraMesh.generate_mesh_from_vertices(coord,enod)

unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
sgname = 'P3221' # Quartz
phases = [Phase(unit_cell, sgname)]
B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )
euler_angles = np.random.rand(mesh.number_of_elements, 3) * 2 * np.pi
eU = np.array( [tools.euler_to_u(ea[0], ea[1], ea[2]) for ea in euler_angles] )
ephase = np.zeros((mesh.number_of_elements,)).astype(int)
polycrystal = Polycrystal(mesh, ephase, eU, eB, phases)

w = detector_size*2 # full field beam
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
xray_propagation_direction = np.array([1,0,0]) * 2 * np.pi / wavelength
polarization_vector = np.array([0,1,0])
beam = Beam(beam_vertices, xray_propagation_direction, wavelength, polarization_vector)

rotation_angle = 5*np.pi/180.
rotation_axis = np.array([0,1,0])
translation = np.array([0,0,0])
motion  = RigidBodyMotion(rotation_axis, rotation_angle, translation)

polycrystal.diffract( beam, detector, motion )
pixim = detector.render(frame_number=0, method='project')

import matplotlib.pyplot as plt
plt.imshow( pixim>0 , cmap='gray')
plt.title("Hits: "+str(len(detector.frames[0]) ))
plt.show()
