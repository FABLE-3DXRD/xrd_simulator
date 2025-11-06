'''Script to create slab sample components to be used within the xrd_simulator package. 
They are stored by default in the /artifacts/samples/ folder within the current directory'''

import os
from xrd_simulator.mesh import TetraMesh
import phases
from scipy.spatial.transform import Rotation as R
from xrd_simulator.polycrystal import Polycrystal
import numpy as np
import pygalmesh
import utilities
import xfab
xfab.CHECKS.activated = False

# Sample parameters.
max_radius_grain = 4. # um
thickness = 100. # um
height = 200. # um
width = 100. # um
#alias = 'orientation_icecream'
alias = 'simple_powder'
destination = os.path.join('artifacts','samples')
dir_this_file=os.path.dirname(os.path.abspath(__file__))


# def slab_level_set(x):
#     """A level set function for a slab geometry for a defined thickness, width and height.

#     Args:
#         x (_type_): A list representing a point in 3D space

#     Returns:
#         _type_: A float which is negative it the point x is inside the sample, 0 in the border and positive outside
#     """
#     dx = abs(x[0]) - thickness / 2.0
#     dy = abs(x[1]) - width / 2.0
#     dz = abs(x[2]) - height / 2.0
    
#     return max(dx, dy, dz)

# x_ = np.arange(-thickness*0.5,thickness*0.5,max_radius_grain)
# y_ = np.arange(-width*0.5,width*0.5,max_radius_grain)
# z_ = np.arange(-height*0.5,height*0.5,max_radius_grain)
# x, y, z = np.meshgrid(x_, y_, z_)
dimx=int(thickness/max_radius_grain)
dimy=int(width/max_radius_grain)
dimz=int(height/max_radius_grain)

vol = np.uint8(np.ones((dimx,dimy,dimz)))

slab = pygalmesh.generate_from_array(
    vol,
    voxel_size = (max_radius_grain,max_radius_grain,max_radius_grain),
    max_facet_distance=max_radius_grain*2,
    max_cell_circumradius=max_radius_grain*2)
slab.points = slab.points - np.mean(slab.points,axis=0) #center the mesh around 0,0,0
# slab = pygalmesh.generate_mesh(
#     pygalmesh.Cuboid([-thickness*0.5,-width*0.5,-height*0.5],[-thickness*0.5,-width*0.5,-height*0.5]),
#     max_cell_circumradius=max_radius_grain,
#     max_edge_size_at_feature_edges=max_radius_grain*0.3,
#     verbose=False)

mesh = TetraMesh._build_tetramesh(slab)

# mesh = TetraMesh.generate_mesh_from_levelset(
#     level_set=slab_level_set,
#     bounding_radius=np.sqrt((thickness*0.5)**2+(height*0.5)**2+(width*0.5)**2),
#     max_cell_circumradius=max_radius_grain)

# Generate random rotations

orientations = R.random(mesh.number_of_elements).as_matrix()
# Generate a list of random 3D rotation matrices

possible_phases = [phases.a_Ferrite] #,phases.Austenite,phases.Cementite,phases.Oleg,phases.hcp]
chosen_phases = np.zeros(mesh.number_of_elements)
chosen_phases[:] = 0



'''chosen_orientations = np.zeros(mesh.number_of_elements) #utils.select_phases(range(len(possible_phases)),mesh.number_of_elements)

coords = mesh.espherecentroids
chosen_orientations[utilities.interval(coords,(-600,-200),(-2000,2000))] = 0
chosen_orientations[utilities.interval(coords,(-200,200),(-2000,2000))] = 1
chosen_orientations[utilities.interval(coords,(200,600),(-2000,2000))] = 2#chosen_phases[int(mesh.number_of_elements*0.9):] = 1
'''
# chosen_phases[(mesh.espherecentroids[:,0]<0)&(mesh.espherecentroids[:,1]<0)] = 1
# chosen_phases[(mesh.espherecentroids[:,0]<0)&(mesh.espherecentroids[:,1]>=0)] = 2


# chosen_phases[(mesh.espherecentroids[:,1]>-50)&(mesh.espherecentroids[:,1]<=0)] = 3
# chosen_phases[(mesh.espherecentroids[:,1]>0)&(mesh.espherecentroids[:,1]<=50)] = 4

# mask_mix=(mesh.espherecentroids[:,1]>50)


#orientations = possible_orientations[chosen_orientations.astype(int),:,:]

# chosen_phases[utilities.interval(coords,(-7.5,37.5),(45,90))] = 3
# chosen_phases[utilities.interval(coords,(-7.5,37.5),(0,45))] = 4
# chosen_phases[utilities.interval(coords,(-7.5,37.5),(-45,0))] = 0
# chosen_phases[utilities.interval(coords,(-7.5,37.5),(-90,-45))] = 3

# chosen_phases[utilities.interval(coords,(37.5,67.5),(60,90))] = 4
# chosen_phases[utilities.interval(coords,(37.5,67.5),(30,60))] = 0
# chosen_phases[utilities.interval(coords,(37.5,67.5),(0,30))] = 3
# chosen_phases[utilities.interval(coords,(37.5,67.5),(-30,0))] = 3
# chosen_phases[utilities.interval(coords,(37.5,67.5),(-60,-30))] = 4
# chosen_phases[utilities.interval(coords,(37.5,67.5),(-90,-60))] = 0

# chosen_phases[(mesh.espherecentroids[:,1]<=-75)] = 0
# chosen_phases[(mesh.espherecentroids[:,1]>-75)&(mesh.espherecentroids[:,1]<=75)] = 3
# chosen_phases[(mesh.espherecentroids[:,1]>-75)&(mesh.espherecentroids[:,1]<=75)] = 4

# temp=chosen_phases[mask_mix]
# temp[:int(sum(mask_mix)*0.59)]=0
# temp[int(sum(mask_mix)*0.59):int(sum(mask_mix)*0.98)]=3
# temp[int(sum(mask_mix)*0.98):]=4
# chosen_phases[mask_mix]=temp

polycrystal = Polycrystal(mesh,
                          orientations,
                          strain=np.zeros((3, 3)),
                          phases=possible_phases,
                          element_phase_map=list(chosen_phases.astype(int)))

save_file_path = os.path.join(dir_this_file,destination,f'{height}x{width}x{thickness}_{max_radius_grain}um_{alias}')

polycrystal.save(save_file_path, save_mesh_as_xdmf=True)
print(f'File saved in {save_file_path}')



