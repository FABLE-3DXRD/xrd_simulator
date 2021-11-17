import numpy as np
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.detector import Detector
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xfab import tools
from scipy.signal import convolve

np.random.seed(23)

#mesh = TetraMesh.load_mesh_from_file( r"C:\Users\Henningsson\workspace\sandbox\grain_no10.xdmf" ) # nelm 300
mesh = TetraMesh.load_mesh_from_file( r"C:\Users\Henningsson\workspace\sandbox\grain_no9.xdmf" ) # nelm +1000

sample_diameter = 128.

print("")
print('nelm:', mesh.number_of_elements)
print("")

pixel_size = sample_diameter/128.
detector_size = pixel_size*1024
detector_distance = 10 * sample_diameter
d0 = np.array([detector_distance,   -detector_size/2.,  -detector_size/2.])
d1 = np.array([detector_distance,    detector_size/2.,  -detector_size/2.])
d2 = np.array([detector_distance,   -detector_size/2.,   detector_size/2.])

detector = Detector( pixel_size, d0, d1, d2 )

#data = os.path.join( os.path.join(os.path.dirname(__file__), 'data' ), 'Fe_mp-150_conventional_standard.cif' )
unit_cell = [3.64570000, 3.64570000, 3.64570000, 90.0, 90.0, 90.0]
sgname = 'Fm-3m' # Iron
phases = [Phase(unit_cell, sgname)]
B0 = tools.epsilon_to_b( np.zeros((6,)), unit_cell )
eB = np.array( [ B0 for _ in range(mesh.number_of_elements)] )

grain_avg_rot = np.max( [np.radians(1.0), np.random.rand() * 2 * np.pi] )
euler_angles  = grain_avg_rot  + np.random.normal(loc=0.0, scale=np.radians(0.01), size=(mesh.number_of_elements, 3) ) 
eU = np.array( [tools.euler_to_u(ea[0], ea[1], ea[2]) for ea in euler_angles] )
ephase = np.zeros((mesh.number_of_elements,)).astype(int)
polycrystal = Polycrystal(mesh, ephase, eU, eB, phases)

w = detector_size # full field beam
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

rotation_angle = 10.0*np.pi/180.
rotation_axis = np.array([0,0,1])
translation = np.array([0,0,0])
motion  = RigidBodyMotion(rotation_axis, rotation_angle, translation)

polycrystal.diffract( beam, detector, motion )

import cProfile
import pstats
pr = cProfile.Profile()
pr.enable()
pixim = detector.render(frame_number=0, lorentz=False, polarization=False, structure_factor=False)
pr.disable()
pr.dump_stats('tmp_profile_dump')
ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(20)

import matplotlib.pyplot as plt
#pixim[ pixim<=0 ] = 1
#pixim = np.log(pixim)
plt.imshow(pixim , cmap='gray')
plt.title("Hits: "+str(len(detector.frames[0]) ))
plt.show()
