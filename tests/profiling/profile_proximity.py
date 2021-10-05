import numpy as np
from xrd_simulator.beam import Beam
from scipy.spatial import ConvexHull
import cProfile
import pstats

np.random.seed(5)
N = 1000
sphere_centres = (np.random.rand(N, 3)-0.5)*2*10
sphere_radius  = np.random.rand(N,)*10

beam_vertices = np.array([
    [-50., -1., -1. ],
    [-50., -1.,  1. ],
    [-50.,  1.,  1. ],
    [-50.,  1., -1. ],
    [50.,  -1., -1. ],
    [50.,  -1.,  1. ],
    [50.,   1.,  1. ],
    [50.,   1., -1. ]  ])
beam_vertices[:,1:] = beam_vertices[:,1:]/1000. # tiny beam cross section
wavelength = 0.43
k1 = np.array([  1,  0,     0 ])
k2 = np.array([ -1,  0.0001, 0 ])
k1 = (np.pi*2/wavelength)*k1/np.linalg.norm(k1)
k2 = (np.pi*2/wavelength)*k2/np.linalg.norm(k2)
translation = np.array([ 0, 0, 0 ])
beam = Beam(beam_vertices, wavelength, k1, k2, translation)

pr = cProfile.Profile()
pr.enable()
intervals = beam.get_proximity_intervals(sphere_centres, sphere_radius)
pr.disable()
pr.dump_stats('profile_dump')
ps = pstats.Stats('profile_dump').strip_dirs().sort_stats('cumtime')
ps.print_stats(20)

notNone=0
for i,interval in enumerate(intervals):
    if interval[0] is not None:
        notNone+=1
        if 0: print(interval, 's_z=',sphere_centres[i,2],'r=',sphere_radius[i])
print("Computed intervals for "+str(N)+" spheres whereof "+str(notNone)+" had roots")
